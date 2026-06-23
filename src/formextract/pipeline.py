"""End-to-end orchestration: folder of forms -> one CSV row per form.

Flow per page::

    render/normalize -> OSD orient -> [checkbox pass] -> CV preprocess
        -> VLM extraction -> merge checkbox overrides

The checkbox pass runs on the orientation-corrected (not CV-preprocessed)
image, because deskew/CLAHE can shift the spatial anchors it relies on.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from .checkbox import checkbox_results_for_page
from .config import CHECKBOX_FIELDS, FIELD_LIST, SUPPORTED_EXTENSIONS, Settings
from .logging_utils import get_logger
from .parsing import normalize_for_csv, safe_stem
from .preprocessing import preprocess_form_image, save_corrected_page_image
from .vlm import run_qwen_extraction

log = get_logger(__name__)


def _load_prompt(settings: Settings) -> str:
    return Path(settings.prompt_file).read_text(encoding="utf-8")


def process_page_image(
    page_num: int,
    input_img_path: str,
    work_dir: str,
    file_tag: str,
    settings: Settings,
    prompt: str,
) -> dict:
    log.info("Processing page %d for file tag '%s'", page_num, file_tag)

    corrected_img_path = os.path.join(work_dir, f"{file_tag}_page_{page_num}_corrected.png")
    save_corrected_page_image(
        input_img_path, corrected_img_path, enable_osd=settings.enable_osd_orientation
    )

    checkbox_overrides: dict = {}
    if settings.checkbox_enabled:
        try:
            checkbox_overrides = checkbox_results_for_page(
                page_num,
                corrected_img_path,
                file_tag=file_tag,
                outdir=str(settings.checkbox_outdir),
                debug=settings.checkbox_debug,
                model_name=settings.vlm_model,
            )
            log.info("Checkbox overrides for page %d: %s", page_num, checkbox_overrides)
        except Exception as e:  # noqa: BLE001
            log.warning("Checkbox pass failed on page %d: %s", page_num, e)

    model_input_path = corrected_img_path
    if settings.preprocess_enabled:
        try:
            model_input_path = preprocess_form_image(
                model_input_path,
                out_dir=str(settings.preprocess_outdir),
                do_debug=settings.preprocess_debug_images,
                do_clahe=settings.use_clahe,
                do_deskew=settings.do_deskew,
                tag=f"{file_tag}_page_{page_num}",
            )
            log.info("Preprocessed model image: %s", model_input_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Preprocessing failed, using corrected image. Error: %s", e)

    extracted = run_qwen_extraction(
        page_num,
        model_input_path,
        work_dir,
        file_tag,
        model_name=settings.vlm_model,
        prompt=prompt,
        temperature=settings.temperature,
        max_new_tokens=settings.max_new_tokens,
    )

    for k, v in checkbox_overrides.items():
        if k in CHECKBOX_FIELDS:
            extracted[k] = v
    return extracted


def process_pdf(
    file_path: str, work_dir: str, file_tag: str, settings: Settings, prompt: str
) -> dict:
    all_data: dict = {}
    doc = fitz.open(file_path)
    try:
        for i, page in enumerate(doc):
            page_num = i + 1
            pix = page.get_pixmap(dpi=settings.pdf_render_dpi)
            page_img_path = os.path.join(work_dir, f"{file_tag}_page_{page_num}_original.png")
            pix.save(page_img_path)
            all_data.update(
                process_page_image(page_num, page_img_path, work_dir, file_tag, settings, prompt)
            )
    finally:
        doc.close()
    return all_data


def process_image_file(
    file_path: str, work_dir: str, file_tag: str, settings: Settings, prompt: str
) -> dict:
    ext = Path(file_path).suffix.lower()
    normalized_img_path = os.path.join(work_dir, f"{file_tag}_page_1_original.png")
    if ext == ".png":
        shutil.copy2(file_path, normalized_img_path)
    else:
        with Image.open(file_path) as img:
            img.convert("RGB").save(normalized_img_path)
    return process_page_image(1, normalized_img_path, work_dir, file_tag, settings, prompt)


def process_single_form(file_path: str, settings: Settings, prompt: str) -> dict[str, str]:
    """Process one file and return a fully-normalized CSV row dict."""
    file_name = os.path.basename(file_path)
    file_tag = safe_stem(file_name)
    log.info("%s", "=" * 80)
    log.info("Processing file: %s", file_name)

    with tempfile.TemporaryDirectory(prefix=f"fd_{file_tag}_") as work_dir:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            extracted = process_pdf(file_path, work_dir, file_tag, settings, prompt)
        else:
            extracted = process_image_file(file_path, work_dir, file_tag, settings, prompt)

    row = {"source_file": file_name}
    for fld in FIELD_LIST:
        row[fld] = normalize_for_csv(extracted.get(fld, "NA"))

    filled = sum(1 for f in FIELD_LIST if row[f] != "NA")
    log.info("Completed '%s' => filled %d/%d fields", file_name, filled, len(FIELD_LIST))
    return row


def list_input_files(input_folder: str | os.PathLike) -> list[str]:
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found or is not a directory: {input_folder}")
    files = [
        str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda x: os.path.basename(x).lower())
    return files


def write_csv(rows: list[dict], output_csv: str | os.PathLike) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source_file"] + FIELD_LIST
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(settings: Settings) -> list[dict[str, str]]:
    """Process every supported file in the input folder; write the CSV."""
    settings.preprocess_outdir.mkdir(parents=True, exist_ok=True)
    settings.checkbox_outdir.mkdir(parents=True, exist_ok=True)
    prompt = _load_prompt(settings)

    input_files = list_input_files(settings.input_folder)
    if not input_files:
        log.warning("No supported files found in: %s", settings.input_folder)
        return []

    log.info("Found %d supported files in folder: %s", len(input_files), settings.input_folder)

    rows: list[dict[str, str]] = []
    for file_path in input_files:
        try:
            rows.append(process_single_form(file_path, settings, prompt))
        except Exception as e:  # noqa: BLE001
            log.error("Failed to process file '%s': %s", os.path.basename(file_path), e)
            row = {"source_file": os.path.basename(file_path)}
            for fld in FIELD_LIST:
                row[fld] = "NA"
            rows.append(row)

    write_csv(rows, settings.output_csv)
    log.info("Batch data saved to %s (%d rows)", settings.output_csv, len(rows))
    return rows
