"""Download the resume dataset (13,389 resumes, 43 roles, MIT licence) into data/raw/resumes.csv."""
import urllib.request
from pathlib import Path

URL = "https://media.githubusercontent.com/media/noran-mohamed/Resume-Classification-Dataset/main/Dataset.csv"
out = Path(__file__).parent / "data" / "raw" / "resumes.csv"
out.parent.mkdir(parents=True, exist_ok=True)
print(f"Downloading {URL} ...")
urllib.request.urlretrieve(URL, out)
print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
