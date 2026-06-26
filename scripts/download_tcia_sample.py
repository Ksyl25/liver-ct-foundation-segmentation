from pathlib import Path
import json
import shutil
import urllib.parse
import urllib.request
import zipfile

import pydicom


BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"

# Public TCIA collection with real anonymized CT DICOM data.
COLLECTION = "LIDC-IDRI"
PATIENT_ID = "LIDC-IDRI-0001"
MODALITY = "CT"

OUT_DIR = Path("data/raw/dicom/sample_series")
ZIP_PATH = Path("data/raw/dicom/tcia_lidc_sample.zip")
TMP_DIR = Path("data/raw/dicom/_tcia_tmp_extract")


def get_json(endpoint: str, params: dict):
    url = f"{BASE_URL}/{endpoint}?" + urllib.parse.urlencode(params)
    print(f"Querying: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, output_path: Path):
    print(f"Downloading: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response, output_path.open("wb") as f:
        shutil.copyfileobj(response, f)

    print(f"Downloaded to: {output_path.resolve()}")


def is_dicom_file(path: Path) -> bool:
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Query one real CT series from TCIA.
    series = get_json(
        "getSeries",
        {
            "Collection": COLLECTION,
            "PatientID": PATIENT_ID,
            "Modality": MODALITY,
        },
    )

    if not series:
        print("No series found for specific patient. Falling back to first CT series in collection.")
        series = get_json(
            "getSeries",
            {
                "Collection": COLLECTION,
                "Modality": MODALITY,
            },
        )

    if not series:
        raise RuntimeError("No CT series found from TCIA.")

    selected = series[0]
    series_uid = selected["SeriesInstanceUID"]

    print("\nSelected TCIA series:")
    print(f"Collection: {COLLECTION}")
    print(f"PatientID: {selected.get('PatientID', PATIENT_ID)}")
    print(f"Modality: {selected.get('Modality')}")
    print(f"SeriesDescription: {selected.get('SeriesDescription')}")
    print(f"SeriesInstanceUID: {series_uid}")

    # 2. Download the DICOM series as ZIP.
    download_url = f"{BASE_URL}/getImage?" + urllib.parse.urlencode(
        {"SeriesInstanceUID": series_uid}
    )
    download_file(download_url, ZIP_PATH)

    # 3. Extract ZIP.
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting ZIP to: {TMP_DIR.resolve()}")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(TMP_DIR)

    # 4. Keep only DICOM files and copy them into sample_series.
    dicom_files = [p for p in TMP_DIR.rglob("*") if p.is_file() and is_dicom_file(p)]

    if not dicom_files:
        raise RuntimeError("No valid DICOM files found in downloaded ZIP.")

    # Clean previous sample files but keep folder.
    for old_file in OUT_DIR.glob("*"):
        if old_file.is_file() and old_file.name != ".gitkeep":
            old_file.unlink()

    for idx, src in enumerate(sorted(dicom_files), start=1):
        dst = OUT_DIR / f"slice_{idx:04d}.dcm"
        shutil.copy2(src, dst)

    print(f"\nCreated real TCIA DICOM sample series in: {OUT_DIR.resolve()}")
    print(f"DICOM files copied: {len(dicom_files)}")
    print("\nUse this folder in Streamlit DICOM Viewer:")
    print("data/raw/dicom/sample_series")


if __name__ == "__main__":
    main()
