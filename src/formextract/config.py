"""Central configuration.

All settings are environment-driven (prefix ``FORMEXTRACT_``) with sensible
defaults, so the codebase contains **no hardcoded machine-specific paths**.

Examples
--------
Override via environment or a local ``.env`` file::

    FORMEXTRACT_INPUT_FOLDER="/data/firedrills"
    FORMEXTRACT_TESSERACT_CMD="C:/Program Files/Tesseract-OCR/tesseract.exe"
    FORMEXTRACT_PDF_RENDER_DPI=400
"""

from __future__ import annotations

from pathlib import Path

import pytesseract
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical, ordered list of fields extracted per form.
# Order matters: pages are split by index in ``vlm.get_partial_model_for_page``.
FIELD_LIST: list[str] = [
    "Site_Address",
    "DDSO_Provider_Agency",
    "Date",
    "Time_Evacuation_Started",
    "Part_of_Day",
    "Total_time_to_evacuate_to_ground",
    "Total_time_for_all_to_reach_safe_area",
    "Centrally_Monitored_Fire_Alarm_Station",
    "Time_Monitoring_Station_Notified_of_Drill",
    "Time_Monitoring_Station_Reactivated",
    "Time_Monitoring_Station_Received_Alarm",
    "Weather_Conditions",
    "Method_of_Alarm_Activation",
    "Evacuation_Type",
    "Type_of_Evacuation",
    "Location_of_Simulated_Fire",
    "Blocked_Exits_by_Simulated_Fire",
    "Location",
    "Name_of_Individuals_Residing_in_the_Residence",
    "including_away_at_the_Time_of_the_Evacuation",
    "To_Evacuate",
    "At_Safe_Area",
    "Evacuation_Details",
    "Description_of_evacuation",
    "Problems_noted_correction_actions_taken",
    "Did_Evacuation_proceed_in_accordance_with_evac_plan",
    "Were_all_exits_escape_route_clear_of_obstructions",
    "Did_alarms_bells_horns_strobes_function_properly",
    "Did_evacuation_time_meet_location_requirement",
    "was_drill_observed",
]

# Fields whose values come from the dedicated checkbox-localization pass
# (these override the VLM's free-text guess for the same field).
CHECKBOX_FIELDS: set[str] = {
    "Method_of_Alarm_Activation",
    "Type_of_Evacuation",
    "Evacuation_Type",
    "Did_evacuation_time_meet_location_requirement",
}

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class Settings(BaseSettings):
    """Runtime configuration, overridable via env vars or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="FORMEXTRACT_",
        env_file=".env",
        extra="ignore",
    )

    # --- I/O ---------------------------------------------------------------
    input_folder: Path = Field(
        default=Path("data/sample"),
        description="Folder containing form PDFs and/or images to process.",
    )
    output_csv: Path = Field(default=Path("outputs/extracted_firedrill_forms.csv"))

    # --- External tools ----------------------------------------------------
    # Empty string => rely on tesseract being on PATH.
    tesseract_cmd: str = Field(default="")
    vlm_model: str = Field(default="qwen2.5vl:7b")

    # --- Prompt ------------------------------------------------------------
    prompt_file: Path = Field(
        default=Path(__file__).parent / "prompts" / "extraction_prompt.xml",
    )

    # --- Rendering / preprocessing ----------------------------------------
    pdf_render_dpi: int = Field(default=600)
    preprocess_enabled: bool = Field(default=True)
    preprocess_debug_images: bool = Field(default=False)
    preprocess_outdir: Path = Field(default=Path("runs/preprocessed_batch"))
    enable_osd_orientation: bool = Field(default=True)
    use_clahe: bool = Field(default=True)
    do_deskew: bool = Field(default=True)

    # --- Checkbox pass -----------------------------------------------------
    checkbox_enabled: bool = Field(default=True)
    checkbox_debug: bool = Field(default=True)
    checkbox_outdir: Path = Field(default=Path("runs/checkbox_debug_batch"))
    checkbox_ink_ratio_thr: float = Field(default=0.02)

    # --- Decoding ----------------------------------------------------------
    temperature: float = Field(default=0.0)
    max_new_tokens: int = Field(default=512)

    def apply_tesseract_cmd(self) -> None:
        """Point pytesseract at the configured binary, if one was provided."""
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd


def get_settings() -> Settings:
    """Load settings once and apply side effects (tesseract path)."""
    settings = Settings()
    settings.apply_tesseract_cmd()
    return settings
