"""Download the MNIST dataset into data/ as raw IDX files."""

import gzip
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(exist_ok=True)

for name in FILES:
    out = data_dir / name
    if out.exists():
        print(f"{name}: already downloaded")
        continue
    gz = data_dir / f"{name}.gz"
    print(f"{name}: downloading...")
    try:
        urllib.request.urlretrieve(MIRROR + f"{name}.gz", gz)
    except urllib.error.URLError:
        # python.org builds on macOS often lack SSL certificates; curl uses the system ones.
        subprocess.run(["curl", "-fsSL", "-o", str(gz), MIRROR + f"{name}.gz"], check=True)
    with gzip.open(gz, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz.unlink()

print(f"MNIST ready in {data_dir}")
