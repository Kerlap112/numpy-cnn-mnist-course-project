"""Download the official MNIST NumPy archive used by Keras."""

from pathlib import Path
import urllib.request


URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def main() -> None:
    target = Path(__file__).resolve().parents[1] / "data" / "mnist.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"数据集已存在: {target}")
        return
    print(f"正在下载 MNIST: {URL}")
    urllib.request.urlretrieve(URL, target)
    print(f"下载完成: {target} ({target.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
