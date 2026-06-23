"""Generate synthetic, public-safe fire-drill forms + ground-truth labels.

These forms contain **no real data** — every value is randomly drawn from fake
pools. They exist so the demo, smoke tests, and evaluation harness can run on
material that is safe to commit publicly (real PHI never enters the repo).

Output:
    data/sample/synthetic_form_NN.pdf       (2-page PDF per form)
    eval/ground_truth/labels.jsonl          (one JSON record per form)

Usage:
    python -m eval.generate_synthetic_forms --n 12 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
GT_DIR = ROOT / "eval" / "ground_truth"

W, H = 1000, 1294  # ~ US Letter at ~120 DPI

# --- fake value pools --------------------------------------------------------
_STREETS = ["Maple", "Oak", "Birch", "Cedar", "Elm", "Pine", "Willow", "Aspen"]
_AGENCIES = ["Northstar DDSO", "Lakeside Provider Agency", "Hudson Valley DDSO",
             "Greenfield Residential", "Brookline Care Agency"]
_WEATHER = ["Clear", "Rainy", "Cold and windy", "Snow", "Overcast", "Mild"]
_PART_OF_DAY = ["Morning", "Afternoon", "Evening", "Overnight"]
_NAMES = ["J. Rivera", "A. Cole", "M. Singh", "T. Nguyen", "R. Patel", "D. Okafor",
          "S. Brooks", "L. Romano", "K. Adams", "P. Costa"]
_YESNO = ["Yes", "No"]
_ALARM = ["Pull Station", "Smoke Detector", "Other"]
_EVAC_TYPE = ["Full Evacuation to Outside", "Other"]
_TYPE_OF_EVAC = ["Announced", "Unannounced", "Supervised"]


def _font(size: int, *candidates: str) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


LABEL = _font(20, "arialbd.ttf", "DejaVuSans-Bold.ttf")
HAND = _font(22, "segoesc.ttf", "comic.ttf", "Comic Sans MS.ttf", "DejaVuSans-Oblique.ttf")
SMALL = _font(16, "arial.ttf", "DejaVuSans.ttf")


def _make_record(rng: random.Random) -> dict:
    n_res = rng.randint(2, 4)
    residents = rng.sample(_NAMES, n_res)
    alarm = rng.sample(_ALARM, rng.randint(1, 2))
    type_evac = rng.sample(_TYPE_OF_EVAC, rng.randint(1, 2))
    return {
        "Site_Address": f"{rng.randint(10, 990)} {rng.choice(_STREETS)} St",
        "DDSO_Provider_Agency": rng.choice(_AGENCIES),
        "Date": f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/2026",
        "Time_Evacuation_Started": f"{rng.randint(1,12)}:{rng.randint(0,59):02d} {rng.choice(['AM','PM'])}",
        "Part_of_Day": rng.choice(_PART_OF_DAY),
        "Total_time_to_evacuate_to_ground": f"{rng.randint(1,5)} min {rng.randint(0,59)} sec",
        "Total_time_for_all_to_reach_safe_area": f"{rng.randint(2,8)} min",
        "Centrally_Monitored_Fire_Alarm_Station": rng.choice(_YESNO),
        "Time_Monitoring_Station_Notified_of_Drill": f"{rng.randint(1,12)}:{rng.randint(0,59):02d} {rng.choice(['AM','PM'])}",
        "Time_Monitoring_Station_Reactivated": f"{rng.randint(1,12)}:{rng.randint(0,59):02d} {rng.choice(['AM','PM'])}",
        "Time_Monitoring_Station_Received_Alarm": f"{rng.randint(1,12)}:{rng.randint(0,59):02d} {rng.choice(['AM','PM'])}",
        "Weather_Conditions": rng.choice(_WEATHER),
        "Method_of_Alarm_Activation": alarm,
        "Evacuation_Type": rng.choice(_EVAC_TYPE),
        "Type_of_Evacuation": "; ".join(type_evac),
        "Location_of_Simulated_Fire": rng.choice(["Kitchen", "Garage", "Bedroom 2", "Basement", "Hallway"]),
        "Blocked_Exits_by_Simulated_Fire": rng.choice(["Front door", "Side exit", "None", "Rear door"]),
        "Location": rng.choice(["1st floor", "2nd floor", "Whole house"]),
        "Name_of_Individuals_Residing_in_the_Residence": residents,
        "including_away_at_the_Time_of_the_Evacuation": rng.sample(residents, rng.randint(0, 1)) or ["NA"],
        "To_Evacuate": [f"{rng.randint(1,3)}:{rng.randint(0,59):02d}" for _ in residents],
        "At_Safe_Area": [f"{rng.randint(2,5)}:{rng.randint(0,59):02d}" for _ in residents],
        "Evacuation_Details": [rng.choice(["Used main exit", "Assisted", "Independent", "Wheelchair ramp"]) for _ in residents],
        "Description_of_evacuation": rng.choice([
            "All residents evacuated calmly via the main exit.",
            "Staff assisted one resident with mobility needs.",
            "Drill proceeded without incident.",
        ]),
        "Problems_noted_correction_actions_taken": rng.choice([
            "None noted.", "Exit light was dim; bulb replaced.",
            "Resident needed prompting; review plan.",
        ]),
        "Did_Evacuation_proceed_in_accordance_with_evac_plan": rng.choice(_YESNO),
        "Were_all_exits_escape_route_clear_of_obstructions": rng.choice(_YESNO),
        "Did_alarms_bells_horns_strobes_function_properly": rng.choice(_YESNO),
        "Did_evacuation_time_meet_location_requirement": rng.choice(_YESNO),
        "was_drill_observed": rng.choice(_YESNO),
    }


def _checkbox_row(d: ImageDraw.ImageDraw, x: int, y: int, label: str, options: list[str], checked: set[str]) -> None:
    d.text((x, y), label, font=LABEL, fill="black")
    ox = x + 470
    for opt in options:
        d.rectangle([ox, y + 2, ox + 18, y + 20], outline="black", width=2)
        if opt in checked:
            d.line([ox + 2, y + 4, ox + 16, y + 18], fill="black", width=2)
            d.line([ox + 16, y + 4, ox + 2, y + 18], fill="black", width=2)
        d.text((ox + 26, y), opt, font=SMALL, fill="black")
        ox += 30 + int(d.textlength(opt, font=SMALL)) + 30


def _wrap(d: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if d.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _kv(d: ImageDraw.ImageDraw, x: int, y: int, label: str, value: str) -> None:
    d.text((x, y), f"{label}:", font=LABEL, fill="black")
    vx = x + 470
    lines = _wrap(d, value, HAND, max_w=W - vx - 40)
    for i, line in enumerate(lines):
        d.text((vx, y - 2 + i * 26), line, font=HAND, fill=(20, 20, 120))


def _render_page1(rec: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((W // 2 - 220, 30), "RESIDENTIAL FIRE DRILL / EVACUATION RECORD", font=LABEL, fill="black")
    d.line([40, 70, W - 40, 70], fill="black", width=2)
    y = 110
    for label, key in [
        ("Site Address", "Site_Address"),
        ("DDSO / Provider Agency", "DDSO_Provider_Agency"),
        ("Date", "Date"),
        ("Time Evacuation Started", "Time_Evacuation_Started"),
        ("Part of Day", "Part_of_Day"),
        ("Total time to evacuate to ground", "Total_time_to_evacuate_to_ground"),
        ("Total time for all to reach safe area", "Total_time_for_all_to_reach_safe_area"),
        ("Centrally Monitored Fire Alarm Station", "Centrally_Monitored_Fire_Alarm_Station"),
        ("Time Monitoring Station Notified of Drill", "Time_Monitoring_Station_Notified_of_Drill"),
        ("Time Monitoring Station Reactivated", "Time_Monitoring_Station_Reactivated"),
        ("Time Monitoring Station Received Alarm", "Time_Monitoring_Station_Received_Alarm"),
        ("Weather Conditions", "Weather_Conditions"),
    ]:
        _kv(d, 50, y, label, str(rec[key]))
        y += 44

    y += 10
    _checkbox_row(d, 50, y, "Method of Alarm Activation", _ALARM, set(rec["Method_of_Alarm_Activation"]))
    y += 44
    _checkbox_row(d, 50, y, "Evacuation Type", _EVAC_TYPE, {rec["Evacuation_Type"]})
    y += 44
    _checkbox_row(d, 50, y, "Type of Evacuation", _TYPE_OF_EVAC, set(str(rec["Type_of_Evacuation"]).split("; ")))
    y += 54
    for label, key in [
        ("Location of Simulated Fire", "Location_of_Simulated_Fire"),
        ("Blocked Exits by Simulated Fire", "Blocked_Exits_by_Simulated_Fire"),
        ("Location", "Location"),
    ]:
        _kv(d, 50, y, label, str(rec[key]))
        y += 44
    d.text((50, H - 40), "Page 1 of 2  —  SYNTHETIC SAMPLE (no real data)", font=SMALL, fill=(150, 150, 150))
    return img


def _render_page2(rec: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((50, 30), "RESIDENTS & EVACUATION DETAIL", font=LABEL, fill="black")
    d.line([40, 60, W - 40, 60], fill="black", width=2)

    # Resident table
    cols = ["Name", "Away?", "To Evacuate", "At Safe Area", "Details"]
    xs = [50, 300, 400, 560, 720]
    y = 90
    for c, x in zip(cols, xs, strict=True):
        d.text((x, y), c, font=LABEL, fill="black")
    y += 30
    residents = rec["Name_of_Individuals_Residing_in_the_Residence"]
    away = set(rec["including_away_at_the_Time_of_the_Evacuation"])
    for i, name in enumerate(residents):
        d.text((xs[0], y), name, font=HAND, fill=(20, 20, 120))
        d.text((xs[1], y), "Y" if name in away else "N", font=HAND, fill=(20, 20, 120))
        d.text((xs[2], y), rec["To_Evacuate"][i], font=HAND, fill=(20, 20, 120))
        d.text((xs[3], y), rec["At_Safe_Area"][i], font=HAND, fill=(20, 20, 120))
        d.text((xs[4], y), rec["Evacuation_Details"][i], font=HAND, fill=(20, 20, 120))
        y += 38

    y += 30
    _kv(d, 50, y, "Description of evacuation", str(rec["Description_of_evacuation"]))
    y += 60
    _kv(d, 50, y, "Problems noted / actions", str(rec["Problems_noted_correction_actions_taken"]))
    y += 70
    for label, key in [
        ("Did Evacuation proceed per plan", "Did_Evacuation_proceed_in_accordance_with_evac_plan"),
        ("Were all exits/routes clear", "Were_all_exits_escape_route_clear_of_obstructions"),
        ("Did alarms/strobes function", "Did_alarms_bells_horns_strobes_function_properly"),
        ("Was drill observed", "was_drill_observed"),
    ]:
        _kv(d, 50, y, label, str(rec[key]))
        y += 44

    y += 20
    _checkbox_row(d, 50, y, "Did evacuation time meet requirement", _YESNO,
                  {rec["Did_evacuation_time_meet_location_requirement"]})
    d.text((50, H - 40), "Page 2 of 2  —  SYNTHETIC SAMPLE (no real data)", font=SMALL, fill=(150, 150, 150))
    return img


def generate(n: int, seed: int) -> None:
    rng = random.Random(seed)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = GT_DIR / "labels.jsonl"

    with open(labels_path, "w", encoding="utf-8") as gt:
        for i in range(n):
            rec = _make_record(rng)
            src = f"synthetic_form_{i:02d}.pdf"
            p1, p2 = _render_page1(rec), _render_page2(rec)
            p1.save(SAMPLE_DIR / src, save_all=True, append_images=[p2], resolution=120.0)
            gt.write(json.dumps({"source_file": src, "fields": rec}) + "\n")
    print(f"Wrote {n} synthetic forms to {SAMPLE_DIR}")
    print(f"Wrote ground truth to {labels_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=12, help="number of forms to generate")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    generate(args.n, args.seed)


if __name__ == "__main__":
    main()
