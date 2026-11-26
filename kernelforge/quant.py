"""Q8_0 / Q4_0 block quantisation in NumPy, byte-compatible with tinyinfer (and ggml/GGUF).

Q8_0: 32 weights -> fp16 scale d + 32 x int8,           x ~= d * q             (34 bytes/block)
Q4_0: 32 weights -> fp16 scale d + 16 bytes of nibbles, x ~= d * (q - 8)       (18 bytes/block)
      element j (0..15) is the low nibble of qs[j], element j+16 the high nibble.

The rounding rules follow tinyinfer's ``src/quant.c`` exactly (d = amax/127 for Q8_0; signed max /
-8 for Q4_0), so a weight file produced here is consumed unchanged by tinyinfer's kernels.
"""
from __future__ import annotations

import numpy as np

QK = 32
BLOCK_BYTES = {"q8_0": 34, "q4_0": 18}


def quantize_q8_0(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, np.float32)
    assert w.shape[-1] % QK == 0
    b = w.reshape(-1, QK)
    amax = np.abs(b).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    idd = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, 1), 0).astype(np.float32)
    x = (b * idd[:, None]).astype(np.float64)  # float32 product, then exact roundf (half away from 0)
    q = np.sign(x) * np.floor(np.abs(x) + 0.5)
    out = np.zeros((b.shape[0], 34), np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.astype(np.int8).view(np.uint8)
    return out.reshape(*w.shape[:-1], (w.shape[-1] // QK) * 34)


def quantize_q4_0(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, np.float32)
    assert w.shape[-1] % QK == 0
    b = w.reshape(-1, QK)
    am = np.argmax(np.abs(b), axis=1)  # first index of the max |x|, like the C loop with '>'
    maxv = b[np.arange(b.shape[0]), am]
    d = (maxv / np.float32(-8.0)).astype(np.float32)
    idd = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, 1), 0).astype(np.float32)
    x = b * idd[:, None]
    q = np.clip(np.trunc(x + np.float32(8.5)), 0, 15).astype(np.uint8)
    out = np.zeros((b.shape[0], 18), np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.reshape(*w.shape[:-1], (w.shape[-1] // QK) * 18)


def dequantize(data: np.ndarray, fmt: str, shape: tuple) -> np.ndarray:
    nb = int(np.prod(shape)) // QK
    raw = np.ascontiguousarray(data, np.uint8).reshape(nb, BLOCK_BYTES[fmt])
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    if fmt == "q8_0":
        q = raw[:, 2:].view(np.int8).astype(np.float32)
    else:
        qs = raw[:, 2:]
        q = np.concatenate([(qs & 0x0F), (qs >> 4)], axis=1).astype(np.float32) - 8.0
    return (d[:, None] * q).reshape(shape)


def quant_parts(data: np.ndarray, fmt: str, shape: tuple):
    """(scales [rows, nblocks], integer values [rows, cols]) as float64, for blocked oracles."""
    rows, cols = int(np.prod(shape[:-1])), shape[-1]
    nb = cols // QK
    raw = np.ascontiguousarray(data, np.uint8).reshape(rows * nb, BLOCK_BYTES[fmt])
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float64).reshape(rows, nb)
    if fmt == "q8_0":
        q = raw[:, 2:].view(np.int8).astype(np.float64)
    else:
        qs = raw[:, 2:]
        q = np.concatenate([(qs & 0x0F), (qs >> 4)], axis=1).astype(np.float64) - 8.0
    return d, q.reshape(rows, cols)
