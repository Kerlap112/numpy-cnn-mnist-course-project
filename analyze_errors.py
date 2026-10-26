"""Create a visual audit of representative misclassified MNIST samples."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cnn_numpy import TinyCNN, preprocess_images


ROOT = Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> None:
    seed, test_size = 42, 2000
    data = np.load(ROOT / "data" / "mnist.npz")
    order = np.random.default_rng(seed).permutation(len(data["x_test"]))[:test_size]
    raw = data["x_test"][order]
    labels = data["y_test"][order]
    model = TinyCNN.load(ROOT / "artifacts" / "model_weights.npz")
    probs = model.predict_proba(preprocess_images(raw))
    pred = probs.argmax(axis=1)
    errors = np.flatnonzero(pred != labels)
    selected = errors[np.argsort(probs[errors, pred[errors]])[::-1][:6]]

    image = Image.new("RGB", (1140, 460), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "典型误分类样本", fill="#172033", font=font(30, True))
    draw.text((28, 62), "红色表示模型的错误预测，条形图显示置信度最高的三个类别", fill="#667085", font=font(16))
    for col, idx in enumerate(selected):
        x0 = 24 + col * 184
        sample = Image.fromarray(raw[idx]).resize((122, 122), Image.Resampling.NEAREST).convert("RGB")
        image.paste(sample, (x0 + 24, 102))
        draw.rectangle((x0, 92, x0 + 174, 420), outline="#e0e5ec", width=2)
        draw.text((x0 + 16, 240), f"真实 {int(labels[idx])}", fill="#25334a", font=font(18, True))
        draw.text((x0 + 92, 240), f"预测 {int(pred[idx])}", fill="#c33c36", font=font(18, True))
        top = np.argsort(probs[idx])[::-1][:3]
        for row, digit in enumerate(top):
            y = 285 + row * 38
            value = float(probs[idx, digit])
            draw.text((x0 + 16, y), str(int(digit)), fill="#2d3d56", font=font(15, True))
            draw.rectangle((x0 + 38, y + 3, x0 + 148, y + 20), fill="#e9edf3")
            draw.rectangle((x0 + 38, y + 3, x0 + 38 + 110 * value, y + 20), fill="#d74b45" if row == 0 else "#6d8fd4")
            draw.text((x0 + 151, y), f"{value*100:.0f}%", fill="#566176", font=font(13))
    image.save(ROOT / "artifacts" / "misclassified_samples.png")


if __name__ == "__main__":
    main()
