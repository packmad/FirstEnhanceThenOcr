#!/usr/bin/env python3
"""
enhance_ocr.py - Forensic image-enhancement + OCR pipeline
Author: Simone Aonzo, © 2025
Licence: MIT
"""

import argparse, hashlib, json, logging, os, sys, time
from pathlib import Path
from os.path import isfile, join

import cv2
import numpy as np
import pytesseract
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# ------------------------ Helper Functions -----------------------------------

def sha256sum(fp: Path, buf_size: int = 1 << 20) -> str:
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()
    with fp.open("rb") as f:
        while chunk := f.read(buf_size):
            h.update(chunk)
    return h.hexdigest()


def save_step(img: np.ndarray, name: str, out_dir: Path) -> Path:
    """Save image in PNG-lossless format."""
    p = out_dir / f"{name}.png"
    cv2.imwrite(str(p), img)
    return p


def clahe_gray(gray: np.ndarray, clip=3.0, grid=8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def unsharp_mask(img: np.ndarray, ksize=(5, 5), amount=1.5, threshold=0) -> np.ndarray:
    """Simple unsharp mask."""
    blurred = cv2.GaussianBlur(img, ksize, 0)
    sharp = float(amount + 1) * img - float(amount) * blurred
    sharp = np.maximum(sharp, np.zeros(sharp.shape))
    sharp = np.minimum(sharp, 255 * np.ones(sharp.shape))
    sharp = sharp.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(sharp, img, where=low_contrast_mask)
    return sharp


def adaptive_threshold(gray: np.ndarray, block: int = 15, C: int = 8) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, C
    )


def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = binary.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated, angle


def run_ocr(gray_or_binary: np.ndarray, psm: int = 7) -> str:
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(gray_or_binary, config=config).strip()


def init_logger(out_dir: Path) -> None:
    log_fp = out_dir / "pipeline.log"
    logging.basicConfig(
        filename=log_fp,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Forensic enhancement + OCR with full chain-of-custody log"
    )
    ap.add_argument("image", type=Path, help="Path to the evidence image")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"),
                    help="Directory to write results")
    args = ap.parse_args()

    in_path: Path = args.image.resolve()
    out_dir: Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logger(out_dir)

    # ------------- 1. Evidence duplication & hashing ------------------------
    evidence_copy = out_dir / in_path.name
    if evidence_copy != in_path:
        evidence_copy.write_bytes(in_path.read_bytes())
    original_hash = sha256sum(evidence_copy)
    logging.info(f"Original SHA-256: {original_hash}")

    # ------------- 2. Load ---------------------------------------------------
    img_color = cv2.imread(str(evidence_copy))
    if img_color is None:
        logging.error("Could not read image")
        sys.exit(1)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # ------------- 3. CLAHE & Unsharp ---------------------------------------
    gray_clahe = clahe_gray(gray)
    sharp = unsharp_mask(gray_clahe)
    save_step(sharp, "01_clahe_unsharp", out_dir)

    # ------------- 4. Optional Super-resolution -----------------------------
    logging.info("Applying Real-ESRGAN ×2")
    model_opt = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2  # Set to 4 for 4× upscaling
    )
    model_path = join('model', 'RealESRGAN_x2plus.pth')
    assert isfile(model_path)
    model = RealESRGANer(scale=2, device="cpu", model_path=model_path, model=model_opt)
    sr_img, _ = model.enhance(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))
    sharp = cv2.cvtColor(np.array(sr_img), cv2.COLOR_RGB2GRAY)
    save_step(sharp, "02_super_res", out_dir)

    # ------------- 5. Adaptive threshold & morphology -----------------------
    thresh = adaptive_threshold(sharp)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_step(cleaned, "03_thresh_clean", out_dir)

    # ------------- 6. Deskew -------------------------------------------------
    deskewed, angle = deskew(cleaned)
    save_step(deskewed, "04_deskew", out_dir)
    logging.info(f"Deskew angle: {angle:+.2f}°")

    # ------------- 7. OCR ----------------------------------------------------
    text = run_ocr(deskewed)
    (out_dir / "05_ocr.txt").write_text(text, encoding="utf-8")
    logging.info(f"OCR result:\n{text}")

    # ------------- 8. Chain-of-custody log ----------------------------------
    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_file": str(in_path),
        "working_copy": str(evidence_copy),
        "original_sha256": original_hash,
        "opencv_version": cv2.__version__,
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "filters": [
            {"name": "CLAHE", "clipLimit": 3.0, "tileGrid": 8},
            {"name": "UnsharpMask", "ksize": [5, 5], "amount": 1.5},
        ],
    }
    meta["filters"].append({"name": "Real-ESRGAN", "scale": 2})
    meta["filters"].extend(
        [
            {"name": "AdaptiveThreshold", "block": 15, "C": 8},
            {"name": "MorphOpen", "kernel": [3, 3]},
            {"name": "Deskew", "angle": angle},
            {"name": "TesseractOCR", "oem": 3, "psm": 7},
        ]
    )
    with (out_dir / "chain_of_custody.json").open("w") as f:
        json.dump(meta, f, indent=2)

    logging.info("Finished. All artefacts saved to %s", out_dir)


if __name__ == "__main__":
    main()
