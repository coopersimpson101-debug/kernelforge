"""Q8_0/Q4_0 must be byte-compatible with tinyinfer (and ggml)."""
import numpy as np

from kernelforge import quant


def test_block_sizes_and_layout():
    w = np.arange(64, dtype=np.float32).reshape(1, 64) - 31.5
    q8 = quant.quantize_q8_0(w)
    q4 = quant.quantize_q4_0(w)
    assert q8.shape == (1, 68) and q4.shape == (1, 36)
    # first Q8_0 block: scale = amax/127 as little-endian fp16, then int8 values
    d = np.frombuffer(q8[0, :2].tobytes(), np.float16)[0]
    assert d == np.float16(np.float32(31.5) / np.float32(127))
    assert np.frombuffer(q8[0, 2:34].tobytes(), np.int8)[0] == -127


def test_q4_nibble_order():
    w = np.zeros((1, 32), np.float32)
    w[0, 0], w[0, 16] = -8.0, 4.0  # signed max is -8 -> d = 1, q = x + 8
    q4 = quant.quantize_q4_0(w)
    assert np.frombuffer(q4[0, :2].tobytes(), np.float16)[0] == 1.0
    assert q4[0, 2] == (0 | (12 << 4))  # element 0 = low nibble (0), element 16 = high nibble (12)


def test_error_bounds():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 256)).astype(np.float32)
    for fmt, f, bound in (("q8_0", quant.quantize_q8_0, 0.6 / 127), ("q4_0", quant.quantize_q4_0, 0.13)):
        back = quant.dequantize(f(w), fmt, w.shape)
        amax = np.abs(w.reshape(-1, 32)).max(1, keepdims=True)
        err = np.abs(back - w).reshape(-1, 32)
        # d/2 rounding (Q4_0: up to d at the clamped +amax end), plus fp16 rounding of the scale
        assert np.all(err <= amax * bound + 1e-6), fmt


def test_roundf_half_away_from_zero():
    w = np.array([[127.0, 0.5, -0.5] + [0.0] * 29], np.float32)  # d = 1 -> exact halves
    q = np.frombuffer(quant.quantize_q8_0(w)[0, 2:].tobytes(), np.int8)
    assert q[1] == 1 and q[2] == -1
