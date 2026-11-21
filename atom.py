#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════╗
║           ATOM — Adaptive Tokenization Orbital Model              ║
║  ───────────────────────────────────────────────────────────────  ║
║  Tokenizer : HEPT-Lite  (256 bytes + phase-rotated embeddings)   ║
║  Atención  : Orbital    (ventana local + global anchors)          ║
║  Memoria   : GMS        (Gated Memory Stream entre capas)         ║
║  FFN       : SwiGLU     (más eficiente que ReLU/GELU)            ║
║  Salida    : Byte-level softmax (256 clases)                      ║
║  Hardware  : GPU  (prototipo experimental)                        ║
╚═══════════════════════════════════════════════════════════════════╝

Uso:
    python atom.py               # selección interactiva de dataset
    python atom.py mi_texto.txt  # pasar dataset directamente
"""

import os
import sys
import math
import time
import glob
import logging
import threading
import queue
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gc
import psutil

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("ATOM")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def obtener_uso_ram_mb() -> float:
    """Devuelve la RAM de sistema usada por el proceso en MB."""
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
def obtener_hf_token() -> Optional[str]:
    """Obtiene el token de HF desde entorno, Colab userdata o cache."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from google.colab import userdata
        tok = userdata.get("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder
        tok = HfFolder.get_token()
        if tok:
            return tok
    except Exception:
        pass
    return None


class ATOMConfig:
    def __init__(self):
        # ── Tokenizer ──────────────────────────────────────────────
        self.vocab_size     = 256           # bytes puros, sin BPE

        # ── Arquitectura ───────────────────────────────────────────
        self.dim            = 256           # dimensión oculta
        self.n_layers       = 6             # capas Transformer
        self.n_heads        = 8             # cabezas de atención
        self.ffn_mult       = 4             # multiplicador FFN (SwiGLU)
        self.dropout        = 0.1
        self.max_seq_len    = 512           # longitud máxima en bytes

        # ── Orbital Attention ──────────────────────────────────────
        self.block_size     = 64            # ventana local causal
        self.global_stride  = 8             # posición de cada anchor global

        # ── GMS & Córtex Causal ────────────────────────────────────
        self.gms_enabled    = True          # Causal GMS sin fuga de futuro
        self.cortex_enabled = True          # Córtex Convolucional Causal de Sílabas
        self.mtp_enabled    = True          # Multi-Token Prediction (t+2 DeepSeek style)
        self.n_loops        = 2             # Pasadas recurrentes por las capas (pensamiento iterativo)
        self.adapter_rank   = 16            # Rango bajo de modulación sináptica por ciclo
        self.distill_weight = 0.15          # Auto-destilación metacognitiva Loop 2 -> Loop 1

        # ── Fusión de Checkpoints (Stochastic Weight Averaging / Model Soup)
        self.fusion_enabled = True          # Fusión ponderada por Loss para eliminar ruido
        self.fusion_tau     = 0.08          # Temperatura para softmax de pérdidas
        self.fusion_top_k   = 3             # Número de mejores checkpoints a fusionar

        # ── Entrenamiento ──────────────────────────────────────────
        self.batch_size     = 16
        self.lr             = 3e-4
        self.weight_decay   = 0.01
        self.steps          = 30_000
        self.warmup_steps   = 500
        self.grad_clip      = 1.0
        self.eval_interval  = 50            # cada cuántos pasos evaluar
        self.sample_interval = 50           # cada cuántos pasos mostrar muestra
        self.sample_prompts = [
            "hola",
            "que eres",
            "cuantos parametros tienes?",
            "como te llamas",
            "para que sirves",
            "que es una silaba",
            "2 + 2",
        ]
        self.temp           = 0.75
        self.min_p          = 0.08          # Min-P sampling (elimina basura improbable)
        self.rep_penalty    = 1.30          # Repetition penalty
        self.gen_len        = 100           # bytes a generar en muestras

        # ── Rutas ──────────────────────────────────────────────────
        self.content_dir    = "/content"
        self.drive_dir      = "/content/drive/MyDrive/ATOM/checkpoints"

        # ── Hugging Face ───────────────────────────────────────────
        self.hf_repo        = "alex4567l/atom"
        self.hf_token       = obtener_hf_token()
        self.hf_dir         = "/content/hf_atom"

        self.device         = DEVICE

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────
# RMSNorm (compatible con PyTorch < 2.1)
# ─────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


# ─────────────────────────────────────────────────────────────────────
# HEPT-LITE TOKENIZER
# Byte-level puro. La "magia" de fase ocurre en el embedding del modelo.
# ─────────────────────────────────────────────────────────────────────
class HEPTLiteTokenizer:
    """
    Tokenizador de bytes puro (vocab = 256).
    encode: str  → List[int]  (valores 0-255)
    decode: List[int] → str
    """
    vocab_size = 256

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: List[int]) -> str:
        return bytes(max(0, min(255, i)) for i in ids).decode("utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────
# PHASE-ROTATED BYTE EMBEDDING  (nucleo de HEPT-Lite)
# ─────────────────────────────────────────────────────────────────────
class PhaseByteEmbedding(nn.Module):
    """
    En vez de un embedding estatico normal E[b], cada byte b en {0..255}
    tiene un angulo de fase fijo:  theta(b) = 2*pi*b / 256

    El embedding final combina dos matrices E_real y E_imag mediante
    rotacion compleja:
        real_part = E_real[b]*cos(theta_b) - E_imag[b]*sin(theta_b)
        imag_part = E_real[b]*sin(theta_b) + E_imag[b]*cos(theta_b)
        embed(b)  = concat(real_part, imag_part)   en R^dim

    Resultado:
    - Bytes similares (ej. 'a'=97, 'a con acento'=195) caen geometricamente
      cerca sin ningun entrenamiento previo.
    - El orden esta codificado: rotar A->B != rotar B->A.
    - Solo 256 x dim parametros vs 50257 x dim de BPE.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        self.E_real = nn.Embedding(vocab_size, half)
        self.E_imag = nn.Embedding(vocab_size, half)
        nn.init.normal_(self.E_real.weight, std=0.02)
        nn.init.normal_(self.E_imag.weight, std=0.02)

        # Angulos fijos (no entrenables)
        angles = 2.0 * math.pi * torch.arange(vocab_size) / vocab_size
        self.register_buffer("cos_t", angles.cos())   # (vocab,)
        self.register_buffer("sin_t", angles.sin())   # (vocab,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, T, dim)"""
        er = self.E_real(x)                           # (B, T, dim/2)
        ei = self.E_imag(x)                           # (B, T, dim/2)
        c  = self.cos_t[x].unsqueeze(-1)              # (B, T, 1)
        s  = self.sin_t[x].unsqueeze(-1)              # (B, T, 1)
        real = er * c - ei * s                        # (B, T, dim/2)
        imag = er * s + ei * c                        # (B, T, dim/2)
        return torch.cat([real, imag], dim=-1)        # (B, T, dim)


# ─────────────────────────────────────────────────────────────────────
# ORBITAL ATTENTION MASK
# ─────────────────────────────────────────────────────────────────────
def _make_orbital_mask(seq_len: int, block_size: int,
                       global_stride: int, device: torch.device) -> torch.Tensor:
    """
    Mascara causal orbital:

    LOCAL : cada token i ve los ultimos `block_size` tokens (ventana deslizante).
    GLOBAL: tokens en posiciones multiplo de `global_stride` son "anchors".
            - Un anchor ve TODOS los tokens anteriores a el.
            - TODOS los tokens posteriores pueden ver al anchor
              (informacion de largo alcance sin O(n^2) completo).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

    # 1. Ventana local causal
    for i in range(seq_len):
        start = max(0, i - block_size + 1)
        mask[i, start : i + 1] = 0.0

    # 2. Anchors globales
    for g in range(0, seq_len, global_stride):
        mask[g, : g + 1] = 0.0            # anchor ve todo lo anterior
        if g + 1 < seq_len:
            mask[g + 1 :, g] = 0.0        # todos ven al anchor

    return mask  # (seq_len, seq_len)


# ─────────────────────────────────────────────────────────────────────
# ORBITAL ATTENTION
# ─────────────────────────────────────────────────────────────────────
class OrbitalAttention(nn.Module):
    """
    Multi-head attention con mascara orbital (local + global anchors),
    QK-RMSNorm esférico y temperatura focal aprendible por cabeza.
    """
    def __init__(self, dim: int, n_heads: int,
                 block_size: int, global_stride: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads       = n_heads
        self.head_dim      = dim // n_heads
        self.block_size    = block_size
        self.global_stride = global_stride
        # Escala logarítmica aprendible por cabeza (inicializada en ln(10.0))
        self.logit_scale   = nn.Parameter(torch.ones(1, n_heads, 1, 1) * math.log(10.0))

        self.qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

        # Cache de mascara
        self._mask_cache: Optional[torch.Tensor] = None
        self._mask_len:   int = -1

    def _get_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self._mask_len != T:
            self._mask_cache = _make_orbital_mask(T, self.block_size,
                                                   self.global_stride, device)
            self._mask_len = T
        return self._mask_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        mask = self._get_mask(T, x.device)

        qkv = self.qkv(x)                                     # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = _split(q), _split(k), _split(v)             # (B, H, T, hd)

        # QK-Norm: normalización en hiperesfera unitaria (evita saturación en loops)
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)

        # Escala aprendible por cabeza clamped a 100.0
        scale = self.logit_scale.exp().clamp(max=100.0)
        attn = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────────
# CÓRTEX DE SÍLABAS (Causal Conv1D kernels 3 y 5)
# ─────────────────────────────────────────────────────────────────────
class SyllableCortex(nn.Module):
    """
    Agrupa letras contiguas en representaciones de sílabas/morfemas
    antes de la Atención Orbital mediante convoluciones estrictamente causales.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=3, padding=2)
        self.conv5 = nn.Conv1d(dim, dim, kernel_size=5, padding=4)
        self.proj  = nn.Linear(dim * 2, dim, bias=False)
        self.norm  = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xt = x.transpose(1, 2)
        # Recorte causal estricto: solo pasado y presente
        c3 = F.silu(self.conv3(xt)[:, :, :T])
        c5 = F.silu(self.conv5(xt)[:, :, :T])
        merged = torch.cat([c3, c5], dim=1).transpose(1, 2)
        return self.norm(x + self.proj(merged))


# ─────────────────────────────────────────────────────────────────────
# GATED MEMORY STREAM (GMS) — 100% CAUSAL (Sin fuga del futuro)
# ─────────────────────────────────────────────────────────────────────
class CausalGatedMemoryStream(nn.Module):
    """
    Torrente de memoria estrictamente causal con modulación de sorpresa (Friston Free Energy):
    En cada posición t, modula la huella mnémica según la sorpresa instantánea ||Δx_t||_1
    acumulando causalmente sobre el contexto temporal 0..t (cero clarividencia).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor,
                h_cum: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, dim)
        h_cum : memoria causal acumulada de capas previas
        Devuelve: (bias_causal (B, T, dim), h_cum_nuevo (B, T, dim))
        """
        B, T, D = x.shape

        # Compuerta de Sorpresa instantánea (Friston Free Energy): ||Δx_t||_1
        diff = torch.zeros_like(x)
        diff[:, 1:, :] = torch.abs(x[:, 1:, :] - x[:, :-1, :])
        surprise_weight = torch.sigmoid(diff.mean(dim=-1, keepdim=True))
        weighted_x = x * (1.0 + surprise_weight)

        # Promedio acumulativo causal ponderado por sorpresa: cada token t solo ve hasta t
        denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        causal_hist = weighted_x.cumsum(dim=1) / denom

        if h_cum is not None and h_cum.shape == (B, T, D):
            combined = torch.cat([x, h_cum], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            fused = gate * h_cum + (1.0 - gate) * causal_hist
        else:
            combined = torch.cat([x, causal_hist], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            fused = gate * causal_hist + (1.0 - gate) * x

        bias = self.proj(self.norm(fused))
        return bias, fused


# ─────────────────────────────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────────────────────────────
class SwiGLUFFN(nn.Module):
    """Feed-Forward con activacion SwiGLU."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)   # SwiGLU usa 2/3 del ancho
        self.w1   = nn.Linear(dim, hidden, bias=False)
        self.w2   = nn.Linear(hidden, dim, bias=False)
        self.w3   = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────────────────────────────
# ATOM BLOCK
# ─────────────────────────────────────────────────────────────────────
class ATOMBlock(nn.Module):
    def __init__(self, cfg: ATOMConfig):
        super().__init__()
        self.attn = OrbitalAttention(
            cfg.dim, cfg.n_heads, cfg.block_size,
            cfg.global_stride, cfg.dropout,
        )
        self.ffn  = SwiGLUFFN(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.ln1  = RMSNorm(cfg.dim)
        self.ln2  = RMSNorm(cfg.dim)
        self.gms  = CausalGatedMemoryStream(cfg.dim) if cfg.gms_enabled else None

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Orbital Attention (pre-norm)
        x = x + self.attn(self.ln1(x))

        # 2. Causal Gated Memory Stream
        h_new = h
        if self.gms is not None:
            bias, h_new = self.gms(x, h)
            x = x + bias

        # 3. SwiGLU FFN (pre-norm)
        x = x + self.ffn(self.ln2(x))
        return x, h_new


# ─────────────────────────────────────────────────────────────────────
# SINUSOIDAL POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────
def _make_sinusoidal_pos(max_len: int, dim: int) -> torch.Tensor:
    pe  = torch.zeros(max_len, dim)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ─────────────────────────────────────────────────────────────────────
# LORA-LOOPS: ADAPTADORES CORTICALES POR CICLO
# ─────────────────────────────────────────────────────────────────────
class LoopAdapter(nn.Module):
    """
    Modulador sináptico de bajo rango (Rank-16) no-lineal.
    Especializa cada capa según el ciclo de razonamiento (Loop 1: ortografía, Loop 2: semántica).
    Inicializado en ceros para preservar la identidad en el arranque.
    """
    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up   = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.silu(self.down(x)))


# ─────────────────────────────────────────────────────────────────────
# ATOM MODEL v2.5 (LoRA-Loops + MTP Temporal Fusion + QK-Norm + GMS Causal)
# ─────────────────────────────────────────────────────────────────────
class ATOMModel(nn.Module):
    def __init__(self, cfg: ATOMConfig):
        super().__init__()
        self.cfg = cfg

        # HEPT-Lite: phase-rotated byte embeddings
        self.embed   = PhaseByteEmbedding(cfg.vocab_size, cfg.dim)

        # Córtex Convolucional Causal de Sílabas
        self.cortex  = SyllableCortex(cfg.dim) if getattr(cfg, "cortex_enabled", True) else None

        # Positional encoding sinusoidal fijo
        self.register_buffer("pos_enc", _make_sinusoidal_pos(cfg.max_seq_len, cfg.dim))

        # Capas Transformer orbital
        self.layers  = nn.ModuleList([ATOMBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_out  = RMSNorm(cfg.dim)

        # LoRA-Loops: adaptadores corticales para cada ciclo y capa
        self.n_loops = getattr(cfg, "n_loops", 2)
        rank = getattr(cfg, "adapter_rank", 16)
        self.loop_adapters = nn.ModuleList([
            nn.ModuleList([LoopAdapter(cfg.dim, rank=rank) for _ in range(cfg.n_layers)])
            for _ in range(self.n_loops)
        ])

        # Cabeza de salida t+1 (256 bytes)
        self.head    = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Cabeza MTP auxiliar t+2 (estilo DeepSeek para planificar con anticipación)
        self.head_mtp = nn.Linear(cfg.dim, cfg.vocab_size, bias=False) if getattr(cfg, "mtp_enabled", True) else None

        total = sum(p.numel() for p in self.parameters())
        LOGGER.info(
            f"🔬 ATOM v2.5 | {total/1e6:.2f}M parametros | "
            f"vocab={cfg.vocab_size} bytes | Loops={self.n_loops} | Córtex Sílabas: {'ON' if self.cortex else 'OFF'} | "
            f"MTP: {'ON' if self.head_mtp else 'OFF'} | GMS Causal: {'ON' if cfg.gms_enabled else 'OFF'}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """x: (B, T) -> logits_next: (B, T, 256), logits_mtp: (B, T, 256), logits_loop1: Optional[(B, T, 256)]"""
        B, T = x.shape
        assert T <= self.cfg.max_seq_len, f"T={T} > max_seq_len={self.cfg.max_seq_len}"

        h = self.embed(x) + self.pos_enc[:T].unsqueeze(0)
        if self.cortex is not None:
            h = self.cortex(h)

        mem = None
        h_loop1 = None

        for loop_idx in range(self.n_loops):
            mem = None   # FIX: reset GMS state at the start of each loop so each pass
                         # accumulates its own causal memory instead of inheriting the
                         # deep-layer state of the previous loop (cortical separation).
            for layer_idx, layer in enumerate(self.layers):
                h, mem = layer(h, mem)
                h = self.loop_adapters[loop_idx][layer_idx](h)
            if loop_idx == 0:
                h_loop1 = h

        normed = self.ln_out(h)
        logits_next = self.head(normed)
        logits_mtp  = self.head_mtp(normed) if self.head_mtp is not None else None

        logits_loop1 = None
        if h_loop1 is not None and self.training and self.n_loops > 1:
            logits_loop1 = self.head(self.ln_out(h_loop1))

        return logits_next, logits_mtp, logits_loop1

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new: int = 100,
                 temp: float = 0.75, min_p: float = 0.08, rep_penalty: float = 1.30) -> List[int]:
        """
        Generador con Fusión Temporal MTP (Inferencia Bicéfala) + Min-P + Bloqueo 4-gramas.
        Combina la decisión de t+1 con el plan previo proyectado por t+2 para máxima coherencia.
        """
        self.eval()
        ctx = list(prompt_ids[-self.cfg.max_seq_len:])
        generated: List[int] = []
        prev_mtp_logits: Optional[torch.Tensor] = None

        for _ in range(max_new):
            x = torch.tensor([ctx], dtype=torch.long, device=self.cfg.device)
            logits_next, logits_mtp, *_ = self(x)
            next_logits = (logits_next[0, -1, :].clone()) / max(temp, 1e-6)

            # Inferencia Bicéfala con Fusión Temporal MTP:
            # Combina la predicción actual (t+1) con la visión calculada en el paso previo (t+2)
            if prev_mtp_logits is not None:
                next_logits = next_logits + 0.25 * prev_mtp_logits

            if logits_mtp is not None:
                prev_mtp_logits = (logits_mtp[0, -1, :].clone() / max(temp, 1e-6)).detach()

            # 1. Repetition Penalty dinámico sobre tokens recientes
            recent = ctx[-40:]
            for token_id in set(recent):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= rep_penalty
                else:
                    next_logits[token_id] *= rep_penalty

            # 2. Bloqueo estricto de 4-gramas repetidos (mata bucles infinitos)
            # FIX: precompute Counter en O(T) en vez de O(256*T) por paso de generación
            if len(ctx) >= 4:
                from collections import Counter
                gram_counts: "Counter[tuple]" = Counter(
                    (ctx[j], ctx[j+1], ctx[j+2], ctx[j+3])
                    for j in range(len(ctx) - 3)
                )
                last3 = tuple(ctx[-3:])
                for cand in range(256):
                    if gram_counts[last3 + (cand,)] >= 2:
                        next_logits[cand] = float("-inf")

            # 3. Min-P Sampling
            probs = F.softmax(next_logits, dim=-1)
            p_max = probs.max()
            probs[probs < min_p * p_max] = 0.0
            p_sum = probs.sum()
            if p_sum > 0:
                probs = probs / p_sum
            else:
                probs = F.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probs, 1).item()
            ctx.append(next_id)
            if len(ctx) > self.cfg.max_seq_len:
                ctx = ctx[1:]
            generated.append(next_id)

            # Salida si genera doble salto de línea
            if len(generated) >= 4 and generated[-2:] == [10, 10]:
                break

        self.train()
        return generated


# ─────────────────────────────────────────────────────────────────────
# BOUNDARY-FOCAL LOSS (Castigo a inicios de palabra + MTP + Auto-Destilación)
# ─────────────────────────────────────────────────────────────────────
def boundary_focal_loss(logits_next: torch.Tensor, targets_next: torch.Tensor,
                        inputs: torch.Tensor,
                        logits_mtp: Optional[torch.Tensor] = None,
                        targets_mtp: Optional[torch.Tensor] = None,
                        logits_loop1: Optional[torch.Tensor] = None,
                        gamma: float = 1.5,
                        distill_weight: float = 0.15) -> torch.Tensor:
    """
    Penaliza fuertemente el error en los inicios de palabra (tras espacio o \n)
    con focal scaling (1 - p)^gamma, predice t+2 (MTP) y auto-destila el ciclo rápido.
    """
    B, T, V = logits_next.shape
    ce = F.cross_entropy(logits_next.view(-1, V), targets_next.view(-1), reduction="none")
    p = torch.exp(-ce.clamp(max=20.0))
    focal = (1.0 - p) ** gamma

    # Inicios de palabra: el byte de entrada anterior fue espacio (32) o newline (10)
    is_boundary = (inputs == 32) | (inputs == 10)
    boundary_w = torch.where(is_boundary.view(-1), 2.5, 1.0)
    total_loss = (ce * focal * boundary_w).mean()

    # MTP Loss (t+2)
    if logits_mtp is not None and targets_mtp is not None:
        ce_mtp = F.cross_entropy(logits_mtp.view(-1, V), targets_mtp.view(-1))
        total_loss = total_loss + 0.35 * ce_mtp

    # Auto-Destilación Metacognitiva: Loop 2 (profundo) enseña a Loop 1 (rápido)
    if logits_loop1 is not None and distill_weight > 0.0:
        loss_distill = F.kl_div(
            F.log_softmax(logits_loop1.view(-1, V), dim=-1),
            F.softmax(logits_next.detach().view(-1, V), dim=-1),
            reduction="batchmean"
        )
        total_loss = total_loss + distill_weight * loss_distill

    return total_loss


# ─────────────────────────────────────────────────────────────────────
# DATASETS (Memmap con alineación de palabra)
# ─────────────────────────────────────────────────────────────────────
class MemmapBatchSampler:
    """Muestreador ultrarrápido alineado al inicio de palabras (0 palabras mutiladas)."""
    def __init__(self, bin_path: str, start_byte: int, end_byte: int,
                 seq_len: int, batch_size: int, device: str):
        self.bin_path   = bin_path
        self.start      = start_byte
        self.end        = end_byte
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.device     = device
        self.max_start  = max(1, (end_byte - start_byte) - seq_len - 64)
        self.mmap       = np.memmap(bin_path, dtype=np.uint8, mode="r")

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extrae un batch alineado donde cada secuencia empieza tras espacio o newline.
        Devuelve (x, y_next, y_mtp)
        """
        raw_offsets = self.start + np.random.randint(0, self.max_start, size=self.batch_size)
        rows = []
        needed = self.seq_len + 2
        for off in raw_offsets:
            shift = 0
            while shift < 32 and (off + shift + needed) < self.end:
                b = self.mmap[off + shift]
                if b == 32 or b == 10:  # espacio o salto de línea
                    shift += 1
                    break
                shift += 1
            start_idx = off + shift
            rows.append(self.mmap[start_idx : start_idx + needed])

        batch = torch.from_numpy(np.stack(rows).astype(np.int64))
        x     = batch[:, :-2].to(self.device, non_blocking=True)
        y_nxt = batch[:, 1:-1].to(self.device, non_blocking=True)
        y_mtp = batch[:, 2:].to(self.device, non_blocking=True)
        return x, y_nxt, y_mtp


class ByteDataset(Dataset):
    """Dataset en memoria para archivos pequeños (< 50 MB)."""
    def __init__(self, data: bytes, seq_len: int):
        self.data    = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1].long()
        return chunk[:-1], chunk[1:]


def _calcular_entropia_shannon(chunk: bytes) -> float:
    """Calcula la entropía de Shannon H(X) en bits por byte (0.0 a 8.0)."""
    if len(chunk) < 64:
        return 4.0
    counts = np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(chunk)
    return float(-np.sum(probs * np.log2(probs)))


def construir_binario_memmap(rutas: List[str], bin_path: str) -> int:
    """
    Lee archivos uno a uno y escribe los bytes directamente a disco aplicando
    el Filtro de Densidad de Información de Shannon (3.2 <= H(X) <= 5.5 bits/byte).
    Elimina ruido binario, spam repetitivo y padding con ~0 MB de consumo de RAM.
    """
    if os.path.exists(bin_path) and os.path.getsize(bin_path) > 1024 * 1024:
        total_bytes = os.path.getsize(bin_path)
        LOGGER.info(f"⚡ [MEMMAP] Binario existente en '{bin_path}' ({total_bytes / (1024**2):.1f} MB). Reutilizando...")
        return total_bytes

    total_bytes = 0
    descartados = 0
    LOGGER.info(f"⚡ [MEMMAP] Creando dataset binario purificado con Shannon en '{bin_path}'...")
    os.makedirs(os.path.dirname(os.path.abspath(bin_path)), exist_ok=True)

    # FIX: hoist _escribir_si_valido out of the loop to avoid re-creating
    # a closure on every iteration (closure-in-loop anti-pattern).
    def _escribir_chunk(f_out_ref, datos: bytes, tb_ref: list, desc_ref: list) -> None:
        """Escribe datos al binario solo si su entropía Shannon está en el rango de oro."""
        if not datos:
            return
        h = _calcular_entropia_shannon(datos)
        if 3.2 <= h <= 5.5:
            f_out_ref.write(datos)
            tb_ref[0] += len(datos)
        else:
            desc_ref[0] += len(datos)

    _tb = [0]      # mutable container para evitar nonlocal en función anidada
    _desc = [0]

    with open(bin_path, "wb") as f_out:
        for i, ruta in enumerate(rutas, 1):
            if not os.path.isfile(ruta):
                continue
            ext = Path(ruta).suffix.lower()

            def _escribir_si_valido(datos: bytes, _f=f_out, _tb=_tb, _d=_desc):
                _escribir_chunk(_f, datos, _tb, _d)

            if ext in (".txt", ".py", ".md"):
                with open(ruta, "rb") as f_in:
                    while True:
                        chunk = f_in.read(64 * 1024)
                        if not chunk:
                            break
                        _escribir_si_valido(chunk)
                f_out.write(b"\n")
                total_bytes += 1
            elif ext == ".jsonl":
                with open(ruta, "r", encoding="utf-8", errors="replace") as f_in:
                    for linea in f_in:
                        linea = linea.strip()
                        if not linea:
                            continue
                        try:
                            obj = __import__("json").loads(linea)
                            txt = _extraer_texto_json(obj)
                        except Exception:
                            txt = linea + "\n"
                        chunk = txt.encode("utf-8", errors="replace")
                        _escribir_si_valido(chunk)
            elif ext == ".json":
                try:
                    import json
                    with open(ruta, "r", encoding="utf-8", errors="replace") as f_in:
                        obj = json.load(f_in)
                    txt = _extraer_texto_json(obj)
                    chunk = txt.encode("utf-8", errors="replace")
                    _escribir_si_valido(chunk)
                except Exception as _je:
                    LOGGER.warning(f"Error procesando JSON {ruta}: {_je}")
            else:
                with open(ruta, "rb") as f_in:
                    while True:
                        chunk = f_in.read(64 * 1024)
                        if not chunk:
                            break
                        _escribir_si_valido(chunk)
                f_out.write(b"\n")
                total_bytes += 1

            if i % 25 == 0 or i == len(rutas):
                LOGGER.info(
                    f"  ⚡ [MEMMAP] Procesados {i}/{len(rutas)} archivos "
                    f"({_tb[0] / (1024**2):.1f} MB guardados | {_desc[0] / (1024**2):.1f} MB ruido filtrado)"
                )

    total_bytes = _tb[0]
    descartados = _desc[0]
    LOGGER.info(f"✅ [MEMMAP] Binario completo: {total_bytes / (1024**2):.1f} MB purificados (Ruido descartado: {descartados / (1024**2):.1f} MB).")
    return total_bytes


# ─────────────────────────────────────────────────────────────────────
# HUGGING FACE ASYNC BACKUP (cola de un solo hilo en segundo plano)
# ─────────────────────────────────────────────────────────────────────
_hf_upload_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
_hf_worker_started = False
_hf_worker_lock = threading.Lock()


def _hf_upload_worker() -> None:
    """Worker permanente: consume la cola de backups uno a uno sin pausar GPU."""
    while True:
        item = _hf_upload_queue.get()
        if item is None:
            break
        checkpoint_path, hf_repo, hf_token = item
        fname = os.path.basename(checkpoint_path)
        try:
            if not os.path.exists(checkpoint_path):
                LOGGER.warning(f"    ⚠️  [HF BACKUP] Archivo no existe: {fname}")
                continue
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            LOGGER.info(f"    ⬆️  [HF BACKUP ▶] Subiendo {fname} a {hf_repo}/checkpoints/...")
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"checkpoints/{fname}",
                repo_id=hf_repo,
                repo_type="dataset",
                commit_message=f"Auto-backup checkpoint {fname}",
            )
            LOGGER.info(f"    ☁️  [HF BACKUP ✅] {fname} sincronizado en {hf_repo}/checkpoints/")
        except Exception as _e:
            LOGGER.warning(f"    ⚠️  [HF BACKUP] Error subiendo {fname}: {_e}")
        finally:
            _hf_upload_queue.task_done()


def respaldar_checkpoint_en_hf_async(checkpoint_path: str, hf_repo: str, hf_token: str) -> None:
    """Encola un respaldo en Hugging Face sin bloquear el entrenamiento."""
    global _hf_worker_started
    if not hf_token or not hf_repo or not os.path.exists(checkpoint_path):
        return

    with _hf_worker_lock:
        if not _hf_worker_started:
            t = threading.Thread(target=_hf_upload_worker, daemon=True, name="ATOM-HF-Backup")
            t.start()
            _hf_worker_started = True

    pending = _hf_upload_queue.qsize()
    if pending > 0:
        LOGGER.info(f"    🕐 [HF BACKUP] Subida previa activa — {os.path.basename(checkpoint_path)} en cola ({pending + 1}).")
    _hf_upload_queue.put((checkpoint_path, hf_repo, hf_token))


def sincronizar_dataset_hf(hf_repo: str, hf_token: Optional[str], dest_dir: str) -> List[str]:
    """Descarga/sincroniza el dataset desde Hugging Face y devuelve los archivos de texto."""
    os.makedirs(dest_dir, exist_ok=True)
    LOGGER.info(f"☁️  [HUGGINGFACE 🚀] Sincronizando dataset '{hf_repo}' en '{dest_dir}'...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess
        LOGGER.info("[HUGGINGFACE] Instalando huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=hf_repo,
            repo_type="dataset",
            local_dir=dest_dir,
            token=hf_token,
            ignore_patterns=["checkpoints/*", ".git*", "*.pt", "*.bin"],
        )
        LOGGER.info(f"☁️  [HUGGINGFACE ✅] Dataset '{hf_repo}' descargado correctamente.")
    except Exception as e:
        LOGGER.warning(f"⚠️  [HUGGINGFACE] Error al descargar '{hf_repo}': {e}")

    encontrados: List[str] = []
    for patron in _EXTENSIONES:
        encontrados.extend([f for f in glob.glob(os.path.join(dest_dir, patron)) if os.path.isfile(f)])
        encontrados.extend([f for f in glob.glob(os.path.join(dest_dir, "**", patron), recursive=True) if os.path.isfile(f)])
    return sorted(set(encontrados))


# ─────────────────────────────────────────────────────────────────────
# CHECKPOINTS Y FUSIÓN (Softmax Loss Weighting / Model Soup)
# ─────────────────────────────────────────────────────────────────────
def fusionar_state_dicts_ponderados(state_dicts: List[dict], losses: List[float],
                                    tau: float = 0.08) -> Tuple[dict, List[float]]:
    """
    Fusiona K state_dicts mediante ponderación softmax inversa por pérdida de validación:
    w_i = exp(-Loss_i / tau) / sum_j exp(-Loss_j / tau)
    W_ATOM = sum_i w_i * x_i
    
    El modelo con menor pérdida domina la convergencia, mientras que los demás
    cancelan el ruido estocástico del gradiente sobre los parámetros.
    """
    losses_arr = np.array(losses, dtype=np.float64)
    z = -losses_arr / max(tau, 1e-6)
    z -= np.max(z)
    exp_z = np.exp(z)
    weights = (exp_z / np.sum(exp_z)).tolist()

    base_sd = state_dicts[0]
    fused_sd = {}
    w0 = float(weights[0])
    for k, v in base_sd.items():
        if torch.is_floating_point(v):
            fused_sd[k] = v.clone().float() * w0
        else:
            fused_sd[k] = v.clone()

    for i in range(1, len(state_dicts)):
        sd_i = state_dicts[i]
        wi = float(weights[i])
        for k, v in sd_i.items():
            if torch.is_floating_point(v):
                fused_sd[k] += v.float() * wi

    for k, v in base_sd.items():
        if torch.is_floating_point(v):
            fused_sd[k] = fused_sd[k].to(dtype=v.dtype)

    return fused_sd, weights


def actualizar_pool_y_fusionar(model: ATOMModel, optimizer: torch.optim.Optimizer,
                               step: int, val_loss: float, cfg: ATOMConfig) -> Optional[float]:
    """
    Mantiene los Top-K mejores checkpoints en Drive y produce 'atom_fused.pt'
    desruidizado con la fórmula ponderada de Boltzmann/Softmax.
    """
    if not getattr(cfg, "fusion_enabled", True):
        return None

    top_k = getattr(cfg, "fusion_top_k", 3)
    tau = getattr(cfg, "fusion_tau", 0.08)
    os.makedirs(cfg.drive_dir, exist_ok=True)
    meta_path = os.path.join(cfg.drive_dir, "checkpoints_pool.json")

    pool = []
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                pool = json.load(f)
        except Exception:
            pool = []

    # Validar archivos existentes
    pool = [entry for entry in pool if os.path.exists(entry.get("path", ""))]

    # Guardar este checkpoint con nombre descriptivo
    nuevo_nombre = f"atom_step_{step}_val_{val_loss:.4f}.pt"
    nuevo_path = os.path.join(cfg.drive_dir, nuevo_nombre)
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "val_loss":  val_loss,
        "config":    cfg.as_dict(),
    }, nuevo_path)

    pool.append({"path": nuevo_path, "step": step, "val_loss": val_loss})
    # Ordenar por menor val_loss
    pool.sort(key=lambda x: x["val_loss"])

    # Poda de checkpoints sobrantes para no saturar almacenamiento
    while len(pool) > top_k:
        descartado = pool.pop(-1)
        if os.path.exists(descartado.get("path", "")):
            try:
                os.remove(descartado["path"])
            except Exception:
                pass

    try:
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(pool, f, indent=2)
    except Exception:
        pass

    # Si hay al menos 2 checkpoints en el pool, fusionar
    if len(pool) >= 2:
        rutas = [e["path"] for e in pool]
        losses = [e["val_loss"] for e in pool]
        steps = [e["step"] for e in pool]

        sds = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in rutas]
        fused_sd, weights = fusionar_state_dicts_ponderados(sds, losses, tau=tau)

        sep = "═" * 68
        print("\n" + sep)
        print(" 🌀 FUSIÓN DE CHECKPOINTS (DESRUIDIZADOR ESTOCÁSTICO / MODEL SOUP)")
        print(sep)
        print(f"  Fórmula: W_ATOM = sum(w_i * x_i)  |  Tau = {tau:.3f}")
        for i, (st, lv, w) in enumerate(zip(steps, losses, weights), 1):
            print(f"   [{i}] Paso {st:>6d} | Val Loss: {lv:.4f} | Peso: {w*100:6.2f}%")
        print(sep)

        fused_path = os.path.join(cfg.drive_dir, "atom_fused.pt")
        val_est = float(sum(lv * w for lv, w in zip(losses, weights)))
        torch.save({
            "model": fused_sd,
            "step": step,
            "fused_from": pool,
            "weights": weights,
            "val_loss_estimado": val_est,
            "tau": tau,
            "config": cfg.as_dict(),
        }, fused_path)
        LOGGER.info(f"    ✨ [MODEL SOUP] Modelo desruidizado guardado en '{os.path.basename(fused_path)}' (val est: {val_est:.4f})")

        # Respaldo automático asíncrono en Hugging Face
        if getattr(cfg, "hf_repo", None) and getattr(cfg, "hf_token", None):
            respaldar_checkpoint_en_hf_async(fused_path, cfg.hf_repo, cfg.hf_token)

        return val_est

    return None


def ejecutar_fusion_cli(cfg: ATOMConfig, tau: float = 0.08, top_k: int = 3) -> None:
    """Busca checkpoints en Drive y Hugging Face para ejecutar la fusión bajo demanda."""
    LOGGER.info(f"🔍 Buscando checkpoints para fusionar en '{cfg.drive_dir}'...")
    os.makedirs(cfg.drive_dir, exist_ok=True)
    candidatos = glob.glob(os.path.join(cfg.drive_dir, "*.pt"))

    # Intentar descargar de Hugging Face si no hay suficientes
    if len(candidatos) < 2 and getattr(cfg, "hf_repo", None):
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=getattr(cfg, "hf_token", None))
            files = api.list_repo_files(repo_id=cfg.hf_repo, repo_type="dataset")
            pt_files = [f for f in files if f.endswith(".pt")]
            for pf in pt_files:
                hf_hub_download(
                    repo_id=cfg.hf_repo,
                    repo_type="dataset",
                    filename=pf,
                    token=getattr(cfg, "hf_token", None),
                    local_dir=cfg.drive_dir,
                )
            candidatos = glob.glob(os.path.join(cfg.drive_dir, "*.pt"))
        except Exception as _e:
            LOGGER.warning(f"No se pudieron descargar checkpoints adicionales de HF: {_e}")

    # Filtrar solo archivos con state_dict válido y val_loss
    validos = []
    for c in candidatos:
        if os.path.basename(c) in ("atom_fused.pt",):
            continue
        try:
            ckpt = torch.load(c, map_location="cpu", weights_only=False)
            if "model" in ckpt and "val_loss" in ckpt:
                validos.append({
                    "path": c,
                    "step": ckpt.get("step", 0),
                    "val_loss": float(ckpt["val_loss"]),
                })
        except Exception:
            pass

    if len(validos) < 2:
        print(f"❌ Se encontraron {len(validos)} checkpoints válidos. Se necesitan al menos 2 para fusionar.")
        return

    validos.sort(key=lambda x: x["val_loss"])
    seleccionados = validos[:top_k]

    rutas = [e["path"] for e in seleccionados]
    losses = [e["val_loss"] for e in seleccionados]
    steps = [e["step"] for e in seleccionados]

    sds = [torch.load(p, map_location="cpu", weights_only=False)["model"] for p in rutas]
    fused_sd, weights = fusionar_state_dicts_ponderados(sds, losses, tau=tau)

    sep = "═" * 68
    print("\n" + sep)
    print(" 🌀 FUSIÓN DE CHECKPOINTS BAJO DEMANDA (MODEL SOUP)")
    print(sep)
    print(f"  Fórmula: W_ATOM = sum(w_i * x_i)  |  Tau = {tau:.3f}")
    for i, (st, lv, w, r) in enumerate(zip(steps, losses, weights, rutas), 1):
        print(f"   [{i}] {os.path.basename(r):<28} (Paso {st:>5d}) | Val: {lv:.4f} | Peso: {w*100:6.2f}%")
    print(sep)

    fused_path = os.path.join(cfg.drive_dir, "atom_fused.pt")
    torch.save({
        "model": fused_sd,
        "fused_from": seleccionados,
        "weights": weights,
        "val_loss_estimado": float(sum(lv * w for lv, w in zip(losses, weights))),
        "tau": tau,
        "config": cfg.as_dict(),
    }, fused_path)
    print(f"  ✅ [ÉXITO] Checkpoint fusionado guardado en:\n     {fused_path}")
    print(sep + "\n")


def guardar_checkpoint(model: ATOMModel, optimizer: torch.optim.Optimizer,
                       step: int, val_loss: float, cfg: ATOMConfig) -> None:
    os.makedirs(cfg.drive_dir, exist_ok=True)
    path = os.path.join(cfg.drive_dir, "atom_best.pt")
    torch.save({
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step":      step,
        "val_loss":  val_loss,
        "config":    cfg.as_dict(),
    }, path)
    LOGGER.info(f"    💾 [CHECKPOINT] atom_best.pt -> Drive (val={val_loss:.4f})")

    # Respaldo automático asíncrono en Hugging Face
    if getattr(cfg, "hf_repo", None) and getattr(cfg, "hf_token", None):
        respaldar_checkpoint_en_hf_async(path, cfg.hf_repo, cfg.hf_token)

    # Fusión estocástica ponderada por Loss para eliminar ruido
    if getattr(cfg, "fusion_enabled", True):
        actualizar_pool_y_fusionar(model, optimizer, step, val_loss, cfg)


def cargar_checkpoint(model: ATOMModel, optimizer: torch.optim.Optimizer,
                      cfg: ATOMConfig) -> Tuple[int, float]:
    path = os.path.join(cfg.drive_dir, "atom_best.pt")

    # Si no existe en Drive, buscar en Hugging Face
    if not os.path.exists(path) and getattr(cfg, "hf_repo", None):
        try:
            from huggingface_hub import hf_hub_download
            LOGGER.info(f"    ☁️  [HF] Buscando checkpoint atom_best.pt en {cfg.hf_repo}...")
            os.makedirs(cfg.drive_dir, exist_ok=True)
            hf_path = hf_hub_download(
                repo_id=cfg.hf_repo,
                repo_type="dataset",
                filename="checkpoints/atom_best.pt",
                token=getattr(cfg, "hf_token", None),
                local_dir=cfg.drive_dir,
            )
            if os.path.exists(hf_path):
                path = hf_path
                LOGGER.info("    ☁️  [HF ✅] Checkpoint descargado exitosamente desde Hugging Face!")
        except Exception as _hfe:
            LOGGER.info(f"    ℹ️  [HF] No se encontró checkpoint previo en {cfg.hf_repo}")

    if not os.path.exists(path):
        return 0, float("inf")
    ckpt = torch.load(path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception:
        pass
    step = ckpt.get("step", 0)
    val  = ckpt.get("val_loss", float("inf"))
    LOGGER.info(f"    ♻️  [CHECKPOINT] Reanudando desde paso {step} (val={val:.4f})")
    return step, val


# ─────────────────────────────────────────────────────────────────────
# CARGA DE ARCHIVOS  (.txt  .py  .json  .jsonl)
# ─────────────────────────────────────────────────────────────────────
MAX_BYTES = 1 * 1024 * 1024 * 1024   # 1 GB

# Campos que intentamos extraer de JSON/JSONL (en orden de prioridad)
_JSON_TEXT_FIELDS = [
    "text", "content", "instruction", "output", "response",
    "input", "question", "answer", "prompt", "completion",
    "human", "assistant", "value",
]


def _extraer_texto_json(obj) -> str:
    """Extrae texto de un objeto JSON de forma inteligente."""
    if isinstance(obj, str):
        return obj + "\n"

    if isinstance(obj, dict):
        # Buscar campos de texto directos
        for field in _JSON_TEXT_FIELDS:
            if field in obj and isinstance(obj[field], str):
                return obj[field] + "\n"

        # Formato conversations: [{"role": ..., "content": ...}, ...]
        if "conversations" in obj and isinstance(obj["conversations"], list):
            partes = []
            for turn in obj["conversations"]:
                if isinstance(turn, dict):
                    for f in ("content", "value", "text"):
                        if f in turn and isinstance(turn[f], str):
                            partes.append(turn[f])
                            break
            return "\n".join(partes) + "\n"

        # Formato messages (OpenAI-style)
        if "messages" in obj and isinstance(obj["messages"], list):
            partes = []
            for msg in obj["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    partes.append(str(msg["content"]))
            return "\n".join(partes) + "\n"

        # Fallback: concatenar todos los valores string del dict
        partes = [str(v) for v in obj.values() if isinstance(v, str)]
        return "\n".join(partes) + "\n"

    if isinstance(obj, list):
        return "\n".join(_extraer_texto_json(item) for item in obj)

    return ""


def cargar_archivo(path: str) -> str:
    """Lee un archivo y devuelve su contenido como texto plano."""
    if not os.path.isfile(path):
        LOGGER.warning(f"  ⚠️ Saltando directorio o ruta inválida: {path}")
        return ""

    ext = Path(path).suffix.lower()
    LOGGER.info(f"  Leyendo: {os.path.basename(path)}")

    if ext in (".txt", ".py", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".jsonl":
        partes: List[str] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    obj = __import__("json").loads(linea)
                    partes.append(_extraer_texto_json(obj))
                except Exception:
                    partes.append(linea + "\n")
        return "".join(partes)

    elif ext == ".json":
        import json
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        return _extraer_texto_json(obj)

    else:
        # Intentar leer como texto de todos modos
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


# ─────────────────────────────────────────────────────────────────────
# SELECCION INTERACTIVA DE DATASET  (multi-archivo, multi-formato)
# ─────────────────────────────────────────────────────────────────────
_EXTENSIONES = ("*.txt", "*.py", "*.json", "*.jsonl")


def seleccionar_datasets(cfg: ATOMConfig) -> List[str]:
    """
    Muestra todos los archivos soportados en content_dir o descargados de Hugging Face.
    El usuario puede elegir UNO o VARIOS (ej: '1 3 5') separados por espacio,
    o presionar 'H' para sincronizar desde Hugging Face ('alex4567l/atom').
    Limite total: 1 GB.
    """
    content_dir = cfg.content_dir

    def _buscar_archivos():
        encontrados: List[str] = []
        for patron in _EXTENSIONES:
            encontrados.extend([f for f in glob.glob(os.path.join(content_dir, patron)) if os.path.isfile(f)])
            if os.path.exists(cfg.hf_dir):
                encontrados.extend([f for f in glob.glob(os.path.join(cfg.hf_dir, patron)) if os.path.isfile(f)])
                encontrados.extend([f for f in glob.glob(os.path.join(cfg.hf_dir, "**", patron), recursive=True) if os.path.isfile(f)])
        return sorted(set(encontrados))

    archivos = _buscar_archivos()

    print("\n" + "=" * 62)
    print("  DATASETS DISPONIBLES EN /content/")
    print("  Formatos soportados: .txt  .py  .json  .jsonl")
    print("  Puedes elegir varios: ej. '1 3 4'  |  Limite: 1 GB total")
    print(f"  Hugging Face Repo  : {cfg.hf_repo}")
    print("=" * 62)

    if not archivos:
        print(f"  (ningun archivo local encontrado en {content_dir})")
        resp = input(f"  ¿Descargar dataset desde Hugging Face '{cfg.hf_repo}'? [S/n]: ").strip().lower()
        if resp not in ("n", "no"):
            archivos = sincronizar_dataset_hf(cfg.hf_repo, cfg.hf_token, cfg.hf_dir)
        if not archivos:
            ruta = input("  Ruta completa al archivo local: ").strip()
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"No existe: {ruta}")
            return [ruta]

    for i, f in enumerate(archivos, 1):
        kb   = os.path.getsize(f) / 1024
        name = os.path.basename(f)
        ext  = Path(f).suffix.lower()
        print(f"  [{i:2d}] {name:<42} {ext:<7} ({kb:,.1f} KB)")
    print(f"  [ H] ☁️  Descargar/actualizar desde Hugging Face ('{cfg.hf_repo}')")
    print("  [ 0] Escribir ruta manualmente")
    print("=" * 62)

    while True:
        raw = input("  Elige (ej: 1  o  1 3 5  o  H): ").strip()

        if raw.lower() in ("h", "hf"):
            archivos_hf = sincronizar_dataset_hf(cfg.hf_repo, cfg.hf_token, cfg.hf_dir)
            if archivos_hf:
                archivos = _buscar_archivos()
                print("\n  Archivos actualizados tras sincronizar HF:")
                for i, f in enumerate(archivos, 1):
                    kb   = os.path.getsize(f) / 1024
                    name = os.path.basename(f)
                    ext  = Path(f).suffix.lower()
                    print(f"  [{i:2d}] {name:<42} {ext:<7} ({kb:,.1f} KB)")
                continue
            else:
                print("  No se pudieron obtener archivos de HF. Intenta otra opción.")
                continue

        if raw == "0":
            ruta = input("  Ruta completa: ").strip()
            if os.path.exists(ruta):
                return [ruta]
            print(f"  No existe: {ruta}")
            continue

        try:
            indices = [int(x) - 1 for x in raw.split()]
            seleccionados = []
            total_bytes   = 0

            for idx in indices:
                if not (0 <= idx < len(archivos)):
                    print(f"  Numero {idx+1} fuera de rango")
                    seleccionados = []
                    break
                size = os.path.getsize(archivos[idx])
                total_bytes += size
                if total_bytes > MAX_BYTES:
                    print(f"  Limite de 1 GB superado al agregar '{os.path.basename(archivos[idx])}'")
                    print(f"  Total acumulado: {total_bytes/1e9:.2f} GB — elige menos archivos.")
                    seleccionados = []
                    break
                seleccionados.append(archivos[idx])

            if seleccionados:
                print(f"\n  Archivos seleccionados ({total_bytes/1024/1024:.1f} MB total):")
                for f in seleccionados:
                    print(f"    + {os.path.basename(f)}")
                print()
                return seleccionados

        except ValueError:
            pass
        print("  Entrada invalida — escribe numeros separados por espacio o 'H'")


# ─────────────────────────────────────────────────────────────────────
# EVALUACION
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluar(model: ATOMModel, val_sampler: MemmapBatchSampler,
            cfg: ATOMConfig, max_batches: int = 15) -> float:
    model.eval()
    losses: List[float] = []
    for _ in range(max_batches):
        x, y_nxt, y_mtp = val_sampler.get_batch()
        out = model(x)
        logits_nxt = out[0]
        loss   = F.cross_entropy(logits_nxt.view(-1, cfg.vocab_size), y_nxt.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ─────────────────────────────────────────────────────────────────────
# MUESTRAS DEL MODELO Y EN VIVO (Formato estilo Electromagnetism)
# ─────────────────────────────────────────────────────────────────────
def _ppl_quality_label(ppl: float) -> str:
    """Etiqueta de calidad segun perplejidad actual del modelo."""
    if ppl > 300:
        return "Aprendiendo a caminar"
    if ppl > 150:
        return "Reconociendo patrones"
    if ppl > 70:
        return "Formando frases basicas"
    if ppl > 30:
        return "Lenguaje emergente"
    if ppl > 15:
        return "Coherencia media y contexto"
    return "Razonamiento fluido y preciso"


def _format_prompt(p: str) -> str:
    return f"\nPregunta: {p.strip()}\nRespuesta: "


def mostrar_muestra(model: ATOMModel, tokenizer: HEPTLiteTokenizer,
                    cfg: ATOMConfig, step: int, val_loss: float, ppl: float,
                    train_loss: float = 0.0) -> None:
    """Muestra la respuesta más coherente seleccionada (arriba)."""
    p_idx = (step // max(1, cfg.sample_interval)) % len(cfg.sample_prompts)
    raw_prompt = cfg.sample_prompts[p_idx]
    formatted = _format_prompt(raw_prompt)

    ids = tokenizer.encode(formatted)
    out_ids = model.generate(ids, max_new=min(cfg.gen_len, 80), temp=0.70, min_p=0.08)
    full_text = tokenizer.decode(ids + out_ids)
    if formatted in full_text:
        answer = full_text.split(formatted, 1)[-1]
    else:
        answer = tokenizer.decode(out_ids)
    answer = answer.replace("\n", " ").strip()

    quality = _ppl_quality_label(ppl)
    sep = "─" * 70

    print("\n" + sep)
    print(f"MUESTRA DEL MODELO - PASO {step}")
    print(sep)
    print(f"  [Prompt  ] {raw_prompt}")
    print(f"  [Output  ] {answer[:150] if answer else '[generando...]'}")
    print(f"  [Metricas] Val Loss={val_loss:.4f} | PPL={ppl:.2f} | Temp=0.70 | Min-P=0.08")
    print(f"  [Estado  ] {quality}")
    print(sep + "\n")


def generar_respuestas_en_vivo(model: ATOMModel, tokenizer: HEPTLiteTokenizer,
                              cfg: ATOMConfig, step: int, val_loss: float, ppl: float,
                              train_loss: float = 0.0) -> None:
    """Genera y muestra los 7 diálogos en vivo (abajo)."""
    pct = (100.0 * step / cfg.steps) if cfg.steps else 0.0
    sep = "═" * 64

    print(sep)
    print(f" 🗣️  EN VIVO · paso {step}/{cfg.steps} ({pct:.1f}%) · val {val_loss:.4f} · ppl {ppl:.2f} · train {train_loss:.4f}")
    print(sep)
    for raw_prompt in cfg.sample_prompts:
        formatted = _format_prompt(raw_prompt)
        ids = tokenizer.encode(formatted)
        out_ids = model.generate(ids, max_new=min(cfg.gen_len, 70), temp=cfg.temp, min_p=cfg.min_p)
        full_text = tokenizer.decode(ids + out_ids)
        if formatted in full_text:
            out_txt = full_text.split(formatted, 1)[-1]
        else:
            out_txt = tokenizer.decode(out_ids)
        out_txt = out_txt.replace("\n", " ").strip()
        print(f"  Tú: {raw_prompt}")
        print(f"  ATOM: {out_txt[:140] if out_txt else '[...]'}")
    print(sep + "\n")


def configurar_prompts_en_vivo(cfg: ATOMConfig) -> None:
    """Permite al usuario ingresar o personalizar sus 7 prompts antes de entrenar."""
    print("\n" + "═" * 65)
    print("  ✍️  CONFIGURA TUS 7 PROMPTS PARA LAS EVALUACIONES EN VIVO")
    print("  (Escribe tu prompt o solo presiona Enter para usar el sugerido)")
    print("═" * 65)
    nuevos = []
    for i in range(7):
        default_p = cfg.sample_prompts[i] if i < len(cfg.sample_prompts) else f"Prompt {i+1}"
        try:
            p = input(f"  [{i+1}/7] Prompt (Enter para '{default_p}'): ").strip()
            nuevos.append(p if p else default_p)
        except (EOFError, KeyboardInterrupt):
            nuevos.append(default_p)
    cfg.sample_prompts = nuevos
    print("═" * 65)
    print("  ✅ 7 Prompts en vivo listos:")
    for i, p in enumerate(cfg.sample_prompts, 1):
        print(f"     [{i}] {p}")
    print("═" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────
def entrenar(model: ATOMModel, tokenizer: HEPTLiteTokenizer,
             train_sampler: MemmapBatchSampler, val_sampler: MemmapBatchSampler,
             cfg: ATOMConfig) -> None:

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Warmup lineal + cosine decay
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        prog = (step - cfg.warmup_steps) / max(cfg.steps - cfg.warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Reanudar si existe checkpoint en Drive
    start_step, best_val = cargar_checkpoint(model, optimizer, cfg)

    model.train()
    model.to(cfg.device)

    # Banner
    print("\n" + "+" + "-" * 65 + "+")
    print("|" + "  ATOM - ENTRENAMIENTO INICIADO".center(65) + "|")
    print("+" + "-" * 65 + "+")
    rows = [
        f"  Pasos:        {cfg.steps:,}",
        f"  Batch:        {cfg.batch_size}",
        f"  Seq len:      {cfg.max_seq_len} bytes",
        f"  Ventana:      {cfg.block_size} tokens (orbital local)",
        f"  Anchors:      cada {cfg.global_stride} tokens (orbital global)",
        f"  GMS Causal:   {'activado' if cfg.gms_enabled else 'desactivado'}",
        f"  LoRA-Loops:   {getattr(cfg, 'n_loops', 2)} ciclos (Rank {getattr(cfg, 'adapter_rank', 16)})",
        f"  Device:       {cfg.device.upper()}",
        f"  Drive:        {cfg.drive_dir}",
        f"  Hugging Face: {cfg.hf_repo} {'(token detectado ✓)' if cfg.hf_token else '(sin token)'}",
    ]
    for r in rows:
        print("|" + r.ljust(65) + "|")
    print("+" + "-" * 65 + "+\n")
    print(f"{'Paso':>8} | {'Train':>8} | {'Val':>8} | {'PPL':>9} | {'LR':>10}")
    print("-" * 55)

    loss_accum  = 0.0
    t0          = time.time()

    for step in range(start_step, cfg.steps):
        # Muestreo instantáneo directo de disco alineado a palabras
        x, y_nxt, y_mtp = train_sampler.get_batch()

        # Forward + Boundary-Focal Loss + MTP + Auto-Destilación Metacognitiva
        logits_nxt, logits_mtp, logits_loop1 = model(x)
        loss = boundary_focal_loss(logits_nxt, y_nxt, x, logits_mtp, y_mtp,
                                   logits_loop1=logits_loop1,
                                   distill_weight=getattr(cfg, "distill_weight", 0.15))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Centralización de Gradientes (previene colapso neuronal y deriva latente)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None and p.grad.ndim >= 2:
                    p.grad.sub_(p.grad.mean(dim=-1, keepdim=True))

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        loss_accum += loss.item()

        # Progreso visual cada 10 pasos
        if (step + 1) % 10 == 0 and (step + 1) % cfg.eval_interval != 0:
            print(f"  ⚡ Paso {step + 1}/{cfg.steps} | Loss train: {loss.item():.4f}", end="\r")

        # Eval + muestra
        if (step + 1) % cfg.eval_interval == 0:
            avg_train  = loss_accum / cfg.eval_interval
            loss_accum = 0.0

            val_loss   = evaluar(model, val_sampler, cfg)
            ppl        = math.exp(min(val_loss, 20.0))
            lr_now     = optimizer.param_groups[0]["lr"]

            print(f"{step+1:>8} | {avg_train:>8.4f} | {val_loss:>8.4f} | {ppl:>9.2f} | {lr_now:>10.2e}")

            if (step + 1) % cfg.sample_interval == 0:
                mostrar_muestra(model, tokenizer, cfg, step + 1, val_loss, ppl, avg_train)
                generar_respuestas_en_vivo(model, tokenizer, cfg, step + 1, val_loss, ppl, avg_train)

            if val_loss < best_val:
                best_val = val_loss
                guardar_checkpoint(model, optimizer, step + 1, val_loss, cfg)

    elapsed = time.time() - t0
    LOGGER.info(f"Entrenamiento completado en {elapsed/60:.1f} min.")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ATOM - Adaptive Tokenization Orbital Model")
    parser.add_argument("dataset", nargs="?", default=None, help="Ruta a dataset local (.txt, .json, .jsonl, .py)")
    parser.add_argument("--entrenar", action="store_true", help="Iniciar entrenamiento")
    parser.add_argument("--memmap", action="store_true", help="Modo memmap (compatibilidad)")
    parser.add_argument("--hf-dataset", "--database", dest="hf_dataset", type=str, default=None,
                        help="Repositorio del dataset en Hugging Face (ej: alex4567l/atom o alex4567l/electromagnetism)")
    parser.add_argument("--token", "--hf-token", dest="token", type=str, default=None,
                        help="Token de acceso para Hugging Face")
    parser.add_argument("--no-input", action="store_true", help="Omitir solicitud interactiva de prompts")
    parser.add_argument("--steps", type=int, default=None, help="Número total de pasos de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamaño de batch")
    parser.add_argument("--fusionar", action="store_true", help="Fusionar los mejores checkpoints para reducir ruido")
    parser.add_argument("--tau", type=float, default=0.08, help="Temperatura para softmax de pérdidas en la fusión")
    parser.add_argument("--top-k", type=int, default=3, help="Cantidad de mejores checkpoints a fusionar")
    args, unknown = parser.parse_known_args()

    cfg       = ATOMConfig()
    tokenizer = HEPTLiteTokenizer()

    # Sobrescribir config con flags CLI
    if args.token:
        cfg.hf_token = args.token
    if args.hf_dataset:
        cfg.hf_repo = args.hf_dataset
    if args.steps:
        cfg.steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.tau is not None:   # FIX: 'if args.tau:' es falsy para 0.0
        cfg.fusion_tau = args.tau
    if args.top_k is not None:  # FIX: 'if args.top_k:' es falsy para 0
        cfg.fusion_top_k = args.top_k

    # Ejecutar fusión bajo demanda y salir si se solicitó --fusionar
    if getattr(args, "fusionar", False):
        ejecutar_fusion_cli(cfg, tau=cfg.fusion_tau, top_k=cfg.fusion_top_k)
        return

    print("\n" + "=" * 58)
    print("  ATOM - Adaptive Tokenization Orbital Model")
    print("  Prototipo experimental - GPU")
    print("=" * 58)

    # Seleccionar datasets
    if args.dataset and os.path.exists(args.dataset):
        rutas = [args.dataset]
        LOGGER.info(f"Dataset (argumento): {rutas[0]}")
    elif args.hf_dataset:
        LOGGER.info(f"Dataset Hugging Face solicitado: {cfg.hf_repo}")
        rutas = sincronizar_dataset_hf(cfg.hf_repo, cfg.hf_token, cfg.hf_dir)
        if not rutas:
            LOGGER.warning("⚠️ No se encontraron archivos tras la descarga de HF. Abriendo selector:")
            rutas = seleccionar_datasets(cfg)
    else:
        rutas = seleccionar_datasets(cfg)

    # ── Construir dataset en disco con Memmap (Consumo RAM: 0 MB) ──
    bin_dir  = "/content" if os.path.exists("/content") else cfg.content_dir
    bin_path = os.path.join(bin_dir, "atom_dataset.bin")

    total_bytes = construir_binario_memmap(rutas, bin_path)
    gc.collect()

    if total_bytes <= cfg.max_seq_len:
        raise ValueError(f"El dataset es demasiado pequeño ({total_bytes} bytes).")

    total_mb = total_bytes / (1024 * 1024)
    LOGGER.info(
        f"📊 Dataset binario: {total_bytes:,} bytes ({total_mb:.1f} MB) | "
        f"{total_bytes / cfg.max_seq_len:,.0f} secuencias potenciales"
    )

    # Split 90/10 directamente sobre el binario en disco
    split         = int(0.9 * total_bytes)
    train_sampler = MemmapBatchSampler(bin_path, 0, split, cfg.max_seq_len, cfg.batch_size, cfg.device)
    val_sampler   = MemmapBatchSampler(bin_path, split, total_bytes, cfg.max_seq_len, cfg.batch_size, cfg.device)

    ram_actual = obtener_uso_ram_mb()
    LOGGER.info(f"🗂️  Samplers de disco listos | Batch: {cfg.batch_size} | RAM usada: {ram_actual:.1f} MB (Límite: 12,700 MB) ✓")

    # Configurar los 7 prompts en vivo interactivos
    if not getattr(args, "no_input", False):
        configurar_prompts_en_vivo(cfg)

    # Modelo
    model = ATOMModel(cfg).to(cfg.device)

    # Entrenar
    entrenar(model, tokenizer, train_sampler, val_sampler, cfg)


if __name__ == "__main__":
    main()
