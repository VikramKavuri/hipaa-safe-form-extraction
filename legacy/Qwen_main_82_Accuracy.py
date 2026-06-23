
import os
import re
import json
import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import fitz
import ollama
import pytesseract
import cv2
from PIL import Image
from pydantic import BaseModel, create_model

# =============================================================================
# CONFIG
# =============================================================================
# Folder that contains all firedrill forms (PDFs and/or images)
INPUT_FOLDER = r"C:\Users\tkavuri\Downloads\FireDrills_Folder"

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\tkavuri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Output CSV (one row per file)
OUTPUT_CSV = "extracted_firedrill_forms.csv"

# Prompt file used by Qwen VLM
PROMPT_FILE = "prompt.xml"

# Preprocess for MODEL image (Qwen)
PREPROCESS_ENABLED = True
PREPROCESS_DEBUG_IMAGES = True
PREPROCESS_OUTDIR = "preprocessed_batch"

# Orientation correction (OSD)
ENABLE_TESSERACT_OSD_ORIENTATION = True

# Preprocessing knobs (MODEL image)
USE_CLAHE = True
DO_DESKEW = True

# DPI for PDF rendering
PDF_RENDER_DPI = 600

# Checkbox pass
CHECKBOX_ENABLED = True
CHECKBOX_DEBUG = True
CHECKBOX_OUTDIR = "checkbox_debug_batch"
CHECKBOX_INK_RATIO_THR = 0.02

# Supported file types in folder
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# CSV columns
field_list = [
    "Site_Address", "DDSO_Provider_Agency", "Date", "Time_Evacuation_Started", "Part_of_Day",
    "Total_time_to_evacuate_to_ground", "Total_time_for_all_to_reach_safe_area",
    "Centrally_Monitored_Fire_Alarm_Station", "Time_Monitoring_Station_Notified_of_Drill",
    "Time_Monitoring_Station_Reactivated", "Time_Monitoring_Station_Received_Alarm",
    "Weather_Conditions", "Method_of_Alarm_Activation", "Evacuation_Type", "Type_of_Evacuation",
    "Location_of_Simulated_Fire", "Blocked_Exits_by_Simulated_Fire", "Location",
    "Name_of_Individuals_Residing_in_the_Residence", "including_away_at_the_Time_of_the_Evacuation",
    "To_Evacuate", "At_Safe_Area", "Evacuation_Details", "Description_of_evacuation",
    "Problems_noted_correction_actions_taken", "Did_Evacuation_proceed_in_accordance_with_evac_plan",
    "Were_all_exits_escape_route_clear_of_obstructions", "Did_alarms_bells_horns_strobes_function_properly",
    "Did_evacuation_time_meet_location_requirement", "was_drill_observed"
]

CHECKBOX_FIELDS = {
    "Method_of_Alarm_Activation",
    "Type_of_Evacuation",
    "Evacuation_Type",
    "Did_evacuation_time_meet_location_requirement",
}

with open(PROMPT_FILE, "r", encoding="utf-8") as file:
    xml_prompt = file.read()


# =============================================================================
# Schema
# =============================================================================
class fields(BaseModel):
    Site_Address: str
    DDSO_Provider_Agency: str
    Date: str
    Time_Evacuation_Started: str
    Part_of_Day: str
    Total_time_to_evacuate_to_ground: str
    Total_time_for_all_to_reach_safe_area: str
    Centrally_Monitored_Fire_Alarm_Station: str
    Time_Monitoring_Station_Notified_of_Drill: str
    Time_Monitoring_Station_Reactivated: str
    Time_Monitoring_Station_Received_Alarm: str
    Weather_Conditions: str
    Method_of_Alarm_Activation: list[str]
    Evacuation_Type: str
    Type_of_Evacuation: str
    Location_of_Simulated_Fire: str
    Blocked_Exits_by_Simulated_Fire: str
    Location: str
    Name_of_Individuals_Residing_in_the_Residence: list[str]
    including_away_at_the_Time_of_the_Evacuation: list[str]
    To_Evacuate: list[str]
    At_Safe_Area: list[str]
    Evacuation_Details: list[str]
    Description_of_evacuation: str
    Problems_noted_correction_actions_taken: str
    Did_Evacuation_proceed_in_accordance_with_evac_plan: str
    Were_all_exits_escape_route_clear_of_obstructions: str
    Did_alarms_bells_horns_strobes_function_properly: str
    Did_evacuation_time_meet_location_requirement: str
    was_drill_observed: str


# =============================================================================
# Utility helpers
# =============================================================================
def safe_stem(path_str: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path_str).stem)


def parse_model_json(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] JSON parse failed: {e}")
        print("[WARN] Raw model output (truncated):", response_text[:500])
        return {}


def normalize_for_csv(val):
    if val is None:
        return "NA"
    if isinstance(val, list):
        cleaned = [str(x).strip() for x in val if str(x).strip() and str(x).strip().upper() != "NA"]
        return "; ".join(cleaned) if cleaned else "NA"
    s = str(val).strip()
    return s if s else "NA"


# =============================================================================
# Qwen checkbox helper
# =============================================================================
def qwen_select_checked_options_from_roi(
    roi_image_path: str,
    options: list[str],
    multi_select: bool,
    model_name: str = "qwen2.5vl:7b"
) -> list[str]:
    prompt = f"""
You are a checkbox state classifier.
You will be given ONE cropped ROI image that contains ONE checkbox question row.
ALLOWED OPTIONS (return only from this list): {options}
multi_select = {str(multi_select).lower()}
RULES:
1) Only judge whether each checkbox square is marked.
2) An option is SELECTED only if its checkbox square has a visible mark (X, tick, filled, scribble).
3) Do NOT return all printed option text. Return ONLY selected options from ALLOWED OPTIONS.
4) If no checkbox is marked, return [\"NA\"].
5) If multi_select=false and more than one box is marked, return [\"NA\"] (do not guess).
6) Output MUST be JSON ONLY, matching this schema:
{{\"selected\": [\"...\"]}} or {{\"selected\": [\"NA\"]}}
""".strip()

    schema = {
        "type": "object",
        "properties": {
            "selected": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["selected"]
    }

    try:
        resp = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                    "images": [roi_image_path]
                }
            ],
            format=schema,
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "do_sample": False,
                "max_new_tokens": 200
            }
        )
        txt = resp.get("message", {}).get("content", "").strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
            txt = re.sub(r"\s*```$", "", txt)
        data = json.loads(txt)
        selected = data.get("selected", ["NA"])
        allowed = set(options) | {"NA"}
        selected = [s for s in selected if s in allowed]
        if not selected:
            return ["NA"]
        if not multi_select and selected != ["NA"] and len(selected) != 1:
            return ["NA"]
        return selected
    except Exception as e:
        print(f"[WARN] Qwen checkbox classifier failed for ROI {roi_image_path}: {e}")
        return ["NA"]


# =============================================================================
# Orientation correction via Tesseract OSD
# =============================================================================
def correct_orientation_osd(pil_img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(pil_img)
        m = re.search(r"Rotate:\s+(\d+)", str(osd))
        if not m:
            return pil_img
        rotate = int(m.group(1))
        if rotate in (90, 180, 270):
            corrected = pil_img.rotate(-rotate, expand=True)
            print(f"[INFO] OSD orientation correction applied: rotate {-rotate} degrees")
            return corrected
        return pil_img
    except Exception as e:
        print(f"[WARN] OSD orientation failed: {e}")
        return pil_img


# =============================================================================
# Preprocessing for MODEL image (VLM-friendly)
# =============================================================================
def preprocess_form_image(
    img_path: str,
    out_dir: str = PREPROCESS_OUTDIR,
    do_debug: bool = PREPROCESS_DEBUG_IMAGES,
    do_clahe: bool = USE_CLAHE,
    do_deskew: bool = DO_DESKEW,
    tag: str = ""
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    base = os.path.splitext(os.path.basename(img_path))[0]
    if tag:
        base = f"{tag}_{base}"

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    if do_debug:
        cv2.imwrite(os.path.join(out_dir, f"{base}_step1_gray.png"), gray)

    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        if do_debug:
            cv2.imwrite(os.path.join(out_dir, f"{base}_step3_clahe.png"), gray)

    deskewed = gray
    angle_deg = 0.0

    if do_deskew:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                                minLineLength=200, maxLineGap=10)
        if lines is not None and len(lines) > 0:
            angles = []
            for x1, y1, x2, y2 in lines[:, 0]:
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                ang = np.degrees(np.arctan2(dy, dx))
                if -45 < ang < 45:
                    angles.append(ang)
            if angles:
                angle_deg = float(np.median(angles))
                angle_deg = max(min(angle_deg, 10), -10)
                h, w = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle=-angle_deg, scale=1.0)
                cos = abs(M[0, 0])
                sin = abs(M[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                M[0, 2] += (new_w / 2) - center[0]
                M[1, 2] += (new_h / 2) - center[1]
                deskewed = cv2.warpAffine(gray, M, (new_w, new_h),
                                          flags=cv2.INTER_CUBIC, borderValue=255)
                if do_debug:
                    cv2.imwrite(os.path.join(out_dir, f"{base}_step4_deskewed_{angle_deg:.2f}.png"), deskewed)

    out_path = os.path.join(out_dir, f"{base}_FINAL_preprocessed.png")
    cv2.imwrite(out_path, deskewed)
    return out_path


# =============================================================================
# OCR + checkbox localization helpers
# =============================================================================
def ocr_data(gray_img: np.ndarray, psm: int = 6) -> dict:
    config = f"--oem 1 --psm {psm}"
    return pytesseract.image_to_data(gray_img, output_type=pytesseract.Output.DICT, config=config)


def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def find_line_anchor_bbox(gray_page: np.ndarray, anchor_keywords: list[str], search_band=(0.0, 0.6)):
    H, W = gray_page.shape[:2]
    yb1 = int(H * search_band[0])
    yb2 = int(H * search_band[1])
    band = gray_page[yb1:yb2, :]
    data = ocr_data(band, psm=6)
    n = len(data["text"])

    lines = {}
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)

    keys = [normalize_text(k) for k in anchor_keywords]
    best = None
    for _, idxs in lines.items():
        line_text = " ".join(normalize_text(data["text"][j]) for j in idxs)
        if all(k in line_text for k in keys):
            lefts = [data["left"][j] for j in idxs]
            tops = [data["top"][j] for j in idxs]
            rights = [data["left"][j] + data["width"][j] for j in idxs]
            bottoms = [data["top"][j] + data["height"][j] for j in idxs]
            bbox_band = (min(lefts), min(tops), max(rights), max(bottoms))
            bbox = (bbox_band[0], bbox_band[1] + yb1, bbox_band[2], bbox_band[3] + yb1)
            if best is None or bbox[1] < best[1]:
                best = bbox
    return best


def find_colon_x_in_line(gray_page: np.ndarray, line_bbox: tuple, scan_right_padding: int = 900):
    H, W = gray_page.shape[:2]
    x1, y1, x2, y2 = line_bbox
    pad_y = 8
    yy1 = max(y1 - pad_y, 0)
    yy2 = min(y2 + pad_y, H)
    xx1 = max(x1, 0)
    xx2 = min(x2 + scan_right_padding, W)
    line_roi = gray_page[yy1:yy2, xx1:xx2]

    data = ocr_data(line_roi, psm=7)
    for i, t in enumerate(data["text"]):
        tt = str(t).strip()
        if ":" in tt:
            return xx1 + int(data["left"][i] + data["width"][i] * 0.8)

    config = "--oem 1 --psm 7"
    try:
        boxes = pytesseract.image_to_boxes(line_roi, config=config)
        colon_candidates = []
        for row in boxes.splitlines():
            parts = row.split(" ")
            if len(parts) < 5:
                continue
            ch = parts[0]
            if ch != ":":
                continue
            left = int(parts[1])
            right = int(parts[3])
            cx = (left + right) // 2
            colon_candidates.append(cx)
        if colon_candidates:
            first_colon_x = min(colon_candidates)
            return xx1 + first_colon_x
    except Exception:
        pass
    return None


def build_value_roi_after_colon(gray_page: np.ndarray, line_bbox: tuple, colon_x, roi_height: int = 95,
                                pad_y: int = 25, right_margin: int = 10):
    H, W = gray_page.shape[:2]
    lx1, ly1, lx2, ly2 = line_bbox
    y1 = max(ly1 - pad_y, 0)
    y2 = min(y1 + roi_height, H)

    if colon_x is None:
        colon_x = min(lx2 + 10, W - 1)

    key_bbox = (lx1, y1, min(colon_x + 5, W - 1), y2)
    vx1 = min(colon_x + 2, W - 1)
    vx2 = max(W - right_margin, vx1 + 1)
    value_bbox = (vx1, y1, vx2, y2)
    roi = gray_page[y1:y2, vx1:vx2].copy()
    return roi, value_bbox, key_bbox


def extract_checkbox_group(gray_page: np.ndarray,
                           anchor_keywords: list[str],
                           options: list[str],
                           debug_name: str,
                           multi_select: bool,
                           search_band=(0.05, 0.45)) -> list[str]:
    if CHECKBOX_DEBUG:
        os.makedirs(CHECKBOX_OUTDIR, exist_ok=True)

    label_bbox = find_line_anchor_bbox(gray_page, anchor_keywords, search_band=search_band)
    if label_bbox is None:
        print(f"[WARN] Label not found for keywords: {anchor_keywords}")
        return []

    colon_x = find_colon_x_in_line(gray_page, label_bbox)
    roi, value_bbox, key_bbox = build_value_roi_after_colon(gray_page, label_bbox, colon_x)

    if CHECKBOX_DEBUG:
        overlay = cv2.cvtColor(gray_page.copy(), cv2.COLOR_GRAY2BGR)
        kx1, ky1, kx2, ky2 = key_bbox
        vx1, vy1, vx2, vy2 = value_bbox
        cv2.rectangle(overlay, (kx1, ky1), (kx2, ky2), (0, 255, 0), 2)
        cv2.rectangle(overlay, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(CHECKBOX_OUTDIR, f"{debug_name}_key_value_overlay.png"), overlay)
        cv2.imwrite(os.path.join(CHECKBOX_OUTDIR, f"{debug_name}_value_roi.png"), roi)

    roi_path = os.path.join(CHECKBOX_OUTDIR, f"{debug_name}_value_roi.png")

    # IMPORTANT: preserve caller-provided multi_select (fixed from original code)
    selected = qwen_select_checked_options_from_roi(
        roi_image_path=roi_path,
        options=options,
        multi_select=multi_select,
        model_name="qwen2.5vl:7b"
    )

    if selected == ["NA"]:
        return []
    return selected


def checkbox_results_for_page(page_index: int, page_image_path: str, file_tag: str = "") -> dict:
    results = {}
    bgr = cv2.imread(page_image_path)
    if bgr is None:
        return results

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    if page_index == 1:
        band = (0.05, 0.45)
        sel = extract_checkbox_group(
            gray,
            anchor_keywords=["method", "alarm", "activ"],
            options=["Pull Station", "Smoke Detector", "Other"],
            debug_name=f"{file_tag}_p1_method_alarm",
            multi_select=True,
            search_band=band
        )
        results["Method_of_Alarm_Activation"] = sel if sel else ["NA"]

        sel = extract_checkbox_group(
            gray,
            anchor_keywords=["evacuation", "type"],
            options=["Full Evacuation to Outside", "Other"],
            debug_name=f"{file_tag}_p1_evac_type",
            multi_select=False,
            search_band=band
        )
        results["Evacuation_Type"] = sel[0] if sel else "NA"

        sel = extract_checkbox_group(
            gray,
            anchor_keywords=["type", "evacuation"],
            options=["Announced", "Unannounced", "Supervised"],
            debug_name=f"{file_tag}_p1_type_of_evac",
            multi_select=True,
            search_band=band
        )
        results["Type_of_Evacuation"] = "; ".join(sel) if sel else "NA"

    elif page_index == 2:
        band = (0.65, 0.98)
        sel = extract_checkbox_group(
            gray,
            anchor_keywords=["evac", "time", "meet", "locat", "require"],
            options=["Yes", "No"],
            debug_name=f"{file_tag}_p2_meet_requirement",
            multi_select=False,
            search_band=band
        )
        results["Did_evacuation_time_meet_location_requirement"] = sel[0] if sel else "NA"

    return results


# =============================================================================
# Page processing
# =============================================================================
def save_corrected_page_image(input_img_path: str, output_img_path: str):
    if ENABLE_TESSERACT_OSD_ORIENTATION:
        pil_img = Image.open(input_img_path)
        pil_corr = correct_orientation_osd(pil_img)
        pil_corr.save(output_img_path)
    else:
        shutil.copy2(input_img_path, output_img_path)


def get_partial_model_for_page(page_num: int):
    if page_num == 1:
        page_fields = dict(list(fields.model_fields.items())[:24])
    else:
        page_fields = dict(list(fields.model_fields.items())[24:])

    PartialFields = create_model(
        f"PartialFields_Page_{page_num}",
        **{name: (field.annotation, ...) for name, field in page_fields.items()}
    )
    return PartialFields


def run_qwen_extraction(page_num: int, model_input_path: str, work_dir: str, file_tag: str) -> dict:
    PartialFields = get_partial_model_for_page(page_num)
    print("[INFO] Sending image to Qwen2.5 Vision model...")
    response = ollama.chat(
        model="qwen2.5vl:7b",
        messages=[
            {
                "role": "system",
                "content": str(xml_prompt),
                "images": [model_input_path]
            }
        ],
        format=PartialFields.model_json_schema(),
        options={
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "do_sample": False,
            "max_new_tokens": 512
        }
    )

    model_text = response.get("message", {}).get("content", "").strip()
    print(f"[DEBUG] Output from Model (truncated): {model_text[:500]}")

    with open(os.path.join(work_dir, f"{file_tag}_page_{page_num}_output.json.txt"), "w", encoding="utf-8") as f:
        f.write(model_text)

    extracted_fields = parse_model_json(model_text)
    print(f"[SUCCESS] Extracted {len(extracted_fields)} fields from page {page_num}")
    return extracted_fields


def process_page_image(page_num: int, input_img_path: str, work_dir: str, file_tag: str) -> dict:
    print(f"\n[INFO] Processing Page {page_num} for file tag '{file_tag}'")

    corrected_img_path = os.path.join(work_dir, f"{file_tag}_page_{page_num}_corrected.png")
    save_corrected_page_image(input_img_path, corrected_img_path)

    checkbox_overrides = {}
    if CHECKBOX_ENABLED:
        try:
            checkbox_overrides = checkbox_results_for_page(page_num, corrected_img_path, file_tag=file_tag)
            print(f"[INFO] Checkbox overrides for page {page_num}: {checkbox_overrides}")
        except Exception as e:
            print(f"[WARN] Checkbox pass failed on page {page_num}: {e}")
            checkbox_overrides = {}

    model_input_path = corrected_img_path
    if PREPROCESS_ENABLED:
        try:
            preprocessed_path = preprocess_form_image(
                model_input_path,
                tag=f"{file_tag}_page_{page_num}"
            )
            print("[INFO] Preprocessed model image:", preprocessed_path)
            model_input_path = preprocessed_path
        except Exception as e:
            print(f"[WARN] Preprocessing failed, using corrected/original image. Error: {e}")

    print("[INFO] FINAL model_input_path sent to Qwen:", model_input_path)
    extracted_fields = run_qwen_extraction(page_num, model_input_path, work_dir, file_tag)

    for k, v in checkbox_overrides.items():
        if k in CHECKBOX_FIELDS:
            extracted_fields[k] = v

    return extracted_fields


# =============================================================================
# Single-file processing
# =============================================================================
def process_pdf(file_path: str, work_dir: str, file_tag: str) -> dict:
    all_extracted_data = {}
    doc = fitz.open(file_path)
    try:
        for i, page in enumerate(doc):
            page_num = i + 1
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            page_img_path = os.path.join(work_dir, f"{file_tag}_page_{page_num}_original.png")
            pix.save(page_img_path)
            extracted_fields = process_page_image(page_num, page_img_path, work_dir, file_tag)
            all_extracted_data.update(extracted_fields)
    finally:
        doc.close()
    return all_extracted_data


def process_image_file(file_path: str, work_dir: str, file_tag: str) -> dict:
    all_extracted_data = {}
    ext = Path(file_path).suffix.lower()
    normalized_img_path = os.path.join(work_dir, f"{file_tag}_page_1_original.png")

    if ext == ".png":
        shutil.copy2(file_path, normalized_img_path)
    else:
        with Image.open(file_path) as img:
            img = img.convert("RGB")
            img.save(normalized_img_path)

    extracted_fields = process_page_image(1, normalized_img_path, work_dir, file_tag)
    all_extracted_data.update(extracted_fields)
    return all_extracted_data


def process_single_form(file_path: str) -> dict:
    file_name = os.path.basename(file_path)
    file_tag = safe_stem(file_name)
    print(f"\n{'=' * 90}")
    print(f"[INFO] Processing file: {file_name}")
    print(f"{'=' * 90}")

    with tempfile.TemporaryDirectory(prefix=f"fd_{file_tag}_") as work_dir:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            extracted = process_pdf(file_path, work_dir, file_tag)
        else:
            extracted = process_image_file(file_path, work_dir, file_tag)

    row = {"source_file": file_name}
    for fld in field_list:
        row[fld] = normalize_for_csv(extracted.get(fld, "NA"))

    filled = len([row[f] for f in field_list if row[f] != "NA"])
    print(f"[INFO] Completed '{file_name}' => filled {filled}/{len(field_list)} fields")
    return row


# =============================================================================
# Batch folder processing + CSV creation
# =============================================================================
def list_input_files(input_folder: str) -> list[str]:
    folder = Path(input_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found or is not a directory: {input_folder}")

    files = [str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    files.sort(key=lambda x: os.path.basename(x).lower())
    return files


def write_csv(rows: list[dict], output_csv: str):
    csv_columns = ["source_file"] + field_list
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(PREPROCESS_OUTDIR, exist_ok=True)
    os.makedirs(CHECKBOX_OUTDIR, exist_ok=True)

    input_files = list_input_files(INPUT_FOLDER)
    if not input_files:
        print(f"[WARN] No supported files found in: {INPUT_FOLDER}")
        return

    print(f"[INFO] Found {len(input_files)} supported files in folder: {INPUT_FOLDER}")

    rows = []
    for file_path in input_files:
        try:
            row = process_single_form(file_path)
            rows.append(row)
        except Exception as e:
            print(f"[ERROR] Failed to process file '{os.path.basename(file_path)}': {e}")
            row = {"source_file": os.path.basename(file_path)}
            for fld in field_list:
                row[fld] = "NA"
            rows.append(row)

    write_csv(rows, OUTPUT_CSV)
    print(f"\n[✓] Batch data saved to {OUTPUT_CSV}")
    print(f"[INFO] Total files written to CSV: {len(rows)}")


if __name__ == "__main__":
    main()