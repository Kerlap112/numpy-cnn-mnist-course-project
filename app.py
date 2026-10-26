"""Local browser application for operating the trained classifier."""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cnn_numpy import TinyCNN, preprocess_images


ROOT = Path(__file__).resolve().parents[1]
MODEL = TinyCNN.load(ROOT / "artifacts" / "model_weights.npz")
SAMPLES = np.load(ROOT / "artifacts" / "test_subset.npz")


def image_to_data_url(array: np.ndarray) -> str:
    image = Image.fromarray(array.astype(np.uint8), mode="L").resize((280, 280), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_canvas(data_url: str) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/sample"):
            query = self.path.split("id=", 1)
            idx = int(query[1]) if len(query) == 2 else 0
            idx %= len(SAMPLES["raw"])
            self.send_json(
                {
                    "id": idx,
                    "image": image_to_data_url(SAMPLES["raw"][idx]),
                    "label": int(SAMPLES["labels"][idx]),
                    "prediction": int(SAMPLES["predictions"][idx]),
                }
            )
            return
        if self.path == "/api/metrics":
            metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
            self.send_json(metrics)
            return
        if self.path.startswith("/artifacts/"):
            target = ROOT / self.path.lstrip("/")
            if target.exists():
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        image = decode_canvas(payload["image"])
        x = preprocess_images(image[None, :, :])
        probs = MODEL.predict_proba(x)[0]
        order = np.argsort(probs)[::-1]
        self.send_json(
            {
                "prediction": int(order[0]),
                "confidence": float(probs[order[0]]),
                "probabilities": [float(v) for v in probs],
                "top3": [{"digit": int(i), "probability": float(probs[i])} for i in order[:3]],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"CNN 演示系统已启动: http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
