"""Train the NumPy CNN and write reproducible experiment artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cnn_numpy import TinyCNN, augment_batch, preprocess_images


ROOT = Path(__file__).resolve().parents[1]


def load_mnist(seed: int, train_size: int, val_size: int, test_size: int):
    data = np.load(ROOT / "data" / "mnist.npz")
    x_train_all, y_train_all = data["x_train"], data["y_train"]
    x_test_all, y_test_all = data["x_test"], data["y_test"]
    rng = np.random.default_rng(seed)
    train_order = rng.permutation(len(x_train_all))
    test_order = rng.permutation(len(x_test_all))
    train_idx = train_order[:train_size]
    val_idx = train_order[train_size : train_size + val_size]
    test_idx = test_order[:test_size]
    return (
        preprocess_images(x_train_all[train_idx]),
        y_train_all[train_idx].astype(np.int64),
        preprocess_images(x_train_all[val_idx]),
        y_train_all[val_idx].astype(np.int64),
        preprocess_images(x_test_all[test_idx]),
        y_test_all[test_idx].astype(np.int64),
        x_test_all[test_idx],
    )


def accuracy(model: TinyCNN, x: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(x) == y).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_training_curves(history: list[dict], path: Path) -> None:
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(30, True), _font(20), _font(16)
    draw.text((60, 22), "训练过程曲线", fill="#172033", font=title_font)
    panels = [(70, 100, 555, 470), (645, 100, 1130, 470)]
    for x0, y0, x1, y1 in panels:
        draw.rectangle((x0, y0, x1, y1), outline="#9aa5b1", width=2)
        for k in range(1, 5):
            y = y0 + k * (y1 - y0) / 5
            draw.line((x0, y, x1, y), fill="#e6eaf0", width=1)
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    train_acc = [h["train_accuracy"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]

    def polyline(values, box, color, vmin, vmax):
        x0, y0, x1, y1 = box
        points = []
        for i, value in enumerate(values):
            x = x0 + (i / max(1, len(values) - 1)) * (x1 - x0)
            y = y1 - ((value - vmin) / max(1e-9, vmax - vmin)) * (y1 - y0)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)

    max_loss = max(losses) * 1.08
    polyline(losses, panels[0], "#2f6fed", 0, max_loss)
    polyline(train_acc, panels[1], "#2f6fed", 0.5, 1.0)
    polyline(val_acc, panels[1], "#e05a47", 0.5, 1.0)
    draw.text((70, 68), "交叉熵损失", fill="#172033", font=label_font)
    draw.text((645, 68), "准确率", fill="#172033", font=label_font)
    draw.text((70, 488), "Epoch", fill="#4c566a", font=small_font)
    draw.text((645, 488), "蓝色 训练集   红色 验证集", fill="#4c566a", font=small_font)
    draw.text((80, 112), f"初始 {losses[0]:.3f}", fill="#2f6fed", font=small_font)
    draw.text((400, 430), f"最终 {losses[-1]:.3f}", fill="#2f6fed", font=small_font)
    draw.text((900, 115), f"验证集 {val_acc[-1]*100:.2f}%", fill="#e05a47", font=small_font)
    draw.text((430, 512), f"共 {epochs[-1]} 个 epoch", fill="#697386", font=small_font)
    image.save(path)


def draw_confusion(matrix: np.ndarray, path: Path) -> None:
    cell, left, top = 58, 130, 100
    width, height = left + 10 * cell + 100, top + 10 * cell + 95
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 24), "测试集混淆矩阵", fill="#172033", font=_font(30, True))
    maximum = max(1, int(matrix.max()))
    for i in range(10):
        draw.text((left - 42, top + i * cell + 17), str(i), fill="#273043", font=_font(18, True))
        draw.text((left + i * cell + 22, top - 34), str(i), fill="#273043", font=_font(18, True))
        for j in range(10):
            value = int(matrix[i, j])
            ratio = value / maximum
            color = (int(240 - 160 * ratio), int(246 - 120 * ratio), int(255 - 35 * ratio))
            x0, y0 = left + j * cell, top + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="white")
            fill = "white" if ratio > 0.55 else "#223047"
            text = str(value)
            box = draw.textbbox((0, 0), text, font=_font(14))
            draw.text((x0 + (cell - (box[2] - box[0])) / 2, y0 + 18), text, fill=fill, font=_font(14))
    draw.text((left + 205, height - 48), "预测类别", fill="#4c566a", font=_font(18))
    # Keep the axis caption above the grid so it cannot collide with the
    # middle y-axis tick (the old placement visually read as “真实类别4”).
    draw.text((18, top - 34), "真实类别↓", fill="#4c566a", font=_font(18))
    image.save(path)


def save_prediction_grid(raw: np.ndarray, y: np.ndarray, pred: np.ndarray, path: Path) -> None:
    rows, cols, tile = 4, 6, 120
    image = Image.new("RGB", (cols * tile, rows * tile + 46), "white")
    draw = ImageDraw.Draw(image)
    draw.text((18, 10), "测试样本预测结果", fill="#172033", font=_font(24, True))
    chosen = np.r_[np.flatnonzero(pred == y)[:18], np.flatnonzero(pred != y)[:6]][: rows * cols]
    for k, idx in enumerate(chosen):
        r, c = divmod(k, cols)
        digit = Image.fromarray(raw[idx]).resize((82, 82), Image.Resampling.NEAREST).convert("RGB")
        x, yy = c * tile + 18, r * tile + 56
        image.paste(digit, (x, yy))
        color = "#17804a" if pred[idx] == y[idx] else "#c33c36"
        draw.text((x, yy + 85), f"真 {y[idx]}  预测 {pred[idx]}", fill=color, font=_font(14, True))
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x_train, y_train, x_val, y_val, x_test, y_test, raw_test = load_mnist(
        args.seed, args.train_size, args.val_size, args.test_size
    )
    model = TinyCNN(args.seed)
    optimizer = model.init_adam()
    history: list[dict] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            batch_x = augment_batch(x_train[batch_idx], rng)
            batch_y = y_train[batch_idx]
            logits = model.forward(batch_x, training=True)
            loss, dlogits = model.loss_and_gradient(logits, batch_y)
            grads = model.backward(dlogits)
            model.adam_step(grads, optimizer, args.learning_rate)
            total_loss += loss * len(batch_idx)
            total_seen += len(batch_idx)
        train_acc = accuracy(model, x_train[:4000], y_train[:4000])
        val_acc = accuracy(model, x_val, y_val)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_seen,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | loss={row['train_loss']:.4f} "
            f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time
    output = ROOT / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model_weights.npz")
    test_start = time.perf_counter()
    probs = model.predict_proba(x_test)
    inference_seconds = time.perf_counter() - test_start
    pred = probs.argmax(axis=1)
    test_accuracy = float((pred == y_test).mean())
    matrix = confusion_matrix(y_test, pred)
    per_class = {}
    for digit in range(10):
        mask = y_test == digit
        per_class[str(digit)] = float((pred[mask] == y_test[mask]).mean())
    metrics = {
        "project": "基于NumPy手写实现的轻量级CNN手写数字分类系统",
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam",
        "test_accuracy": test_accuracy,
        "test_error_count": int((pred != y_test).sum()),
        "inference_fps": float(len(x_test) / inference_seconds),
        "single_image_latency_ms": float(inference_seconds / len(x_test) * 1000),
        "training_seconds": elapsed,
        "parameter_count": int(sum(value.size for value in model.params.values())),
        "per_class_accuracy": per_class,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "history": history,
        "confusion_matrix": matrix.tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "test_subset.npz", raw=raw_test[:120], labels=y_test[:120], predictions=pred[:120], probabilities=probs[:120]
    )
    draw_training_curves(history, output / "training_curves.png")
    draw_confusion(matrix, output / "confusion_matrix.png")
    save_prediction_grid(raw_test, y_test, pred, output / "prediction_grid.png")
    print(json.dumps({k: metrics[k] for k in ["test_accuracy", "inference_fps", "training_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
