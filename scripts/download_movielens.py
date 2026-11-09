"""Download MovieLens 100K into data/ml-100k/."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    if (DATA_DIR / "ml-100k" / "u.data").exists():
        print("MovieLens 100K already downloaded")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR / 'ml-100k'}")


if __name__ == "__main__":
    main()
