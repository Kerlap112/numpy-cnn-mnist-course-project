"""A small CNN implemented from first principles with NumPy.

The implementation intentionally avoids deep-learning frameworks.  It contains
the forward and backward passes for convolution, ReLU, max-pooling, a dense
layer, softmax cross entropy, and Adam.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _im2col(x: np.ndarray, kh: int, kw: int) -> tuple[np.ndarray, int, int]:
    n, c, h, w = x.shape
    oh, ow = h - kh + 1, w - kw + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * kh * kw)
    return cols, oh, ow


def _col2im(
    cols: np.ndarray,
    shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    oh: int,
    ow: int,
) -> np.ndarray:
    n, c, h, w = shape
    dx = np.zeros(shape, dtype=np.float32)
    blocks = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 1, 2, 4, 5)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i : i + oh, j : j + ow] += blocks[:, :, :, :, i, j]
    return dx


@dataclass
class AdamState:
    step: int
    m: dict[str, np.ndarray]
    v: dict[str, np.ndarray]


class TinyCNN:
    """Conv(1, 8, 3x3) -> ReLU -> MaxPool(2x2) -> Dense(288, 10)."""

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {
            "conv_w": (rng.normal(0, np.sqrt(2 / 9), size=(8, 1, 3, 3))).astype(np.float32),
            "conv_b": np.zeros(8, dtype=np.float32),
            "fc_w": (rng.normal(0, np.sqrt(2 / 288), size=(288, 10))).astype(np.float32),
            "fc_b": np.zeros(10, dtype=np.float32),
        }
        self.cache: dict[str, np.ndarray | tuple[int, ...] | int] = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        conv_w = self.params["conv_w"]
        cols, oh, ow = _im2col(x, 3, 3)
        conv = cols @ conv_w.reshape(8, -1).T + self.params["conv_b"]
        conv = conv.reshape(x.shape[0], oh, ow, 8).transpose(0, 3, 1, 2)
        relu = np.maximum(conv, 0)

        n, c, h, w = relu.shape
        pooled_blocks = relu.reshape(n, c, h // 2, 2, w // 2, 2)
        pooled = pooled_blocks.max(axis=(3, 5))
        flat = pooled.reshape(n, -1)
        logits = flat @ self.params["fc_w"] + self.params["fc_b"]

        if training:
            self.cache = {
                "x_shape": x.shape,
                "cols": cols,
                "oh": oh,
                "ow": ow,
                "conv": conv,
                "relu": relu,
                "pooled": pooled,
                "flat": flat,
            }
        return logits

    @staticmethod
    def probabilities(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def loss_and_gradient(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        probs = TinyCNN.probabilities(logits)
        n = y.shape[0]
        loss = -np.log(np.clip(probs[np.arange(n), y], 1e-9, 1.0)).mean()
        grad = probs
        grad[np.arange(n), y] -= 1
        grad /= n
        return float(loss), grad

    def backward(self, dlogits: np.ndarray) -> dict[str, np.ndarray]:
        flat = self.cache["flat"]
        pooled = self.cache["pooled"]
        relu = self.cache["relu"]
        conv = self.cache["conv"]
        cols = self.cache["cols"]
        x_shape = self.cache["x_shape"]
        oh = int(self.cache["oh"])
        ow = int(self.cache["ow"])

        assert isinstance(flat, np.ndarray)
        assert isinstance(pooled, np.ndarray)
        assert isinstance(relu, np.ndarray)
        assert isinstance(conv, np.ndarray)
        assert isinstance(cols, np.ndarray)
        assert isinstance(x_shape, tuple)

        grads: dict[str, np.ndarray] = {}
        grads["fc_w"] = flat.T @ dlogits
        grads["fc_b"] = dlogits.sum(axis=0)
        dflat = dlogits @ self.params["fc_w"].T
        dpool = dflat.reshape(pooled.shape)

        n, c, h, w = relu.shape
        blocks = relu.reshape(n, c, h // 2, 2, w // 2, 2)
        maxima = blocks.max(axis=(3, 5), keepdims=True)
        mask = blocks == maxima
        ties = mask.sum(axis=(3, 5), keepdims=True)
        drelu = (mask * dpool[:, :, :, None, :, None] / ties).reshape(relu.shape)
        dconv = drelu * (conv > 0)

        dconv_flat = dconv.transpose(0, 2, 3, 1).reshape(-1, 8)
        grads["conv_w"] = (dconv_flat.T @ cols).reshape(self.params["conv_w"].shape)
        grads["conv_b"] = dconv_flat.sum(axis=0)
        dcols = dconv_flat @ self.params["conv_w"].reshape(8, -1)
        _ = _col2im(dcols, x_shape, 3, 3, oh, ow)
        return grads

    def init_adam(self) -> AdamState:
        return AdamState(
            step=0,
            m={name: np.zeros_like(value) for name, value in self.params.items()},
            v={name: np.zeros_like(value) for name, value in self.params.items()},
        )

    def adam_step(
        self,
        grads: dict[str, np.ndarray],
        state: AdamState,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        state.step += 1
        for name, grad in grads.items():
            grad = np.clip(grad, -5.0, 5.0)
            state.m[name] = beta1 * state.m[name] + (1 - beta1) * grad
            state.v[name] = beta2 * state.v[name] + (1 - beta2) * (grad * grad)
            m_hat = state.m[name] / (1 - beta1**state.step)
            v_hat = state.v[name] / (1 - beta2**state.step)
            self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def predict_proba(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        result = []
        for start in range(0, len(x), batch_size):
            result.append(self.probabilities(self.forward(x[start : start + batch_size])))
        return np.concatenate(result, axis=0)

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self.predict_proba(x, batch_size).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.params)

    @classmethod
    def load(cls, path: str | Path) -> "TinyCNN":
        model = cls()
        data = np.load(path)
        for name in model.params:
            model.params[name] = data[name].astype(np.float32)
        return model


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Normalize 28x28 MNIST images and average-pool them to 14x14."""
    x = images.astype(np.float32) / 255.0
    x = x.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    return x[:, None, :, :]


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random one-pixel translations and light Gaussian noise."""
    result = x.copy()
    shifts = rng.integers(-1, 2, size=(len(x), 2))
    for i, (dy, dx) in enumerate(shifts):
        shifted = np.roll(result[i], (dy, dx), axis=(1, 2))
        if dy > 0:
            shifted[:, :dy, :] = 0
        elif dy < 0:
            shifted[:, dy:, :] = 0
        if dx > 0:
            shifted[:, :, :dx] = 0
        elif dx < 0:
            shifted[:, :, dx:] = 0
        result[i] = shifted
    noise = rng.normal(0, 0.025, size=result.shape).astype(np.float32)
    return np.clip(result + noise, 0, 1)
