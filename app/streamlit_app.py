"""Streamlit demo for the privacy-first document-AI pipeline.

Run with:
    streamlit run app/streamlit_app.py

Two modes:
* **Sample (no model)** — browse synthetic forms, their target structured output,
  and the evaluation report. Runs anywhere with the repo's deps; no Ollama needed.
* **Live (Ollama)** — upload your own form and run the real pipeline end-to-end
  (requires Ollama serving qwen2.5vl:7b and Tesseract installed).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Make the repo root importable so `from app import demo_core` resolves whether this
# file is launched via `streamlit run app/streamlit_app.py` or by Hugging Face Spaces.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app import demo_core

ON_SPACES = bool(os.environ.get("SPACE_ID"))
# "hf" => hosted Qwen2.5-VL via HF Inference Providers (works anywhere, incl. Spaces).
# "ollama" => local model (Live mode only makes sense off-Spaces).
HF_BACKEND = os.environ.get("FORMEXTRACT_VLM_BACKEND", "ollama") == "hf"
LIVE_AVAILABLE = HF_BACKEND or not ON_SPACES
LIVE_LABEL = "Live (hosted Qwen2.5-VL-72B)" if HF_BACKEND else "Live (local Ollama)"

st.set_page_config(page_title="Privacy-First Document AI", page_icon="🔒", layout="wide")

st.title("🔒 Privacy-First Document AI")
st.caption(
    "Structured extraction from handwritten compliance forms — vision-language model "
    "+ classical CV. Production runs 100% locally; this demo can route to a hosted model."
)

modes = ["Sample (no model)"] + ([LIVE_LABEL] if LIVE_AVAILABLE else [])
mode = st.sidebar.radio("Mode", modes)
st.sidebar.markdown(
    "**Sample** browses synthetic, public-safe forms and the eval report — no model "
    "needed.\n\n**Live** extracts fields from a form you upload."
)
if HF_BACKEND:
    st.sidebar.info(
        "**Live mode** routes to **Qwen2.5-VL-72B** via Hugging Face Inference Providers so "
        "it works without local GPUs. The *production* pipeline runs the model fully locally "
        "(HIPAA-safe) — see the [GitHub repo](https://github.com/VikramKavuri/hipaa-safe-form-extraction)."
    )
elif ON_SPACES:
    st.sidebar.info(
        "Running on Hugging Face Spaces — **Sample mode only**. Live extraction needs "
        "a local Ollama + Tesseract; clone the repo and run `streamlit run app/streamlit_app.py`."
    )


def _render_fields(fields: dict) -> None:
    checkbox, other = demo_core.split_fields(fields)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Fields")
        st.json(other)
    with c2:
        st.subheader("Checkbox fields (hybrid classifier)")
        st.json(checkbox)


if mode == "Sample (no model)":
    samples = demo_core.load_samples()
    if not samples:
        st.warning(
            "No synthetic samples found. Generate them first:\n\n"
            "`python -m eval.generate_synthetic_forms --n 12 --seed 7`"
        )
    else:
        names = [s.source_file for s in samples]
        choice = st.sidebar.selectbox("Pick a synthetic form", names)
        sample = next(s for s in samples if s.source_file == choice)

        left, right = st.columns([1, 1])
        with left:
            st.subheader(f"Input — {sample.source_file}")
            try:
                for png in demo_core.render_pdf_pages(sample.pdf_path):
                    st.image(png, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not render PDF: {e}")
        with right:
            st.subheader("Target structured output (ground truth)")
            st.info(
                "This is the labeled ground truth for this synthetic form. In **Live** "
                "mode, the model's prediction appears here instead and is scored against it."
            )
            _render_fields(sample.fields)

    st.divider()
    st.header("📊 Evaluation report")
    if demo_core.REPORT_PLOT.exists():
        st.image(str(demo_core.REPORT_PLOT), caption="Per-field performance")
    md = demo_core.load_report_markdown()
    if md:
        with st.expander("Full metrics table"):
            st.markdown(md)
    else:
        st.caption("Run `python -m eval.run_eval --predictor mock` to generate the report.")

else:  # Live mode
    st.subheader("Upload a form (PDF or image)")
    if HF_BACKEND:
        st.caption(
            "Extraction runs **Qwen2.5-VL-72B** via Hugging Face Inference Providers. "
            "A 2-page form takes ~15–40s."
        )
    else:
        st.warning("Local Live mode requires Ollama serving `qwen2.5vl:7b` and Tesseract.")
    upload = st.file_uploader("Form file", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if upload is not None and st.button("Extract", type="primary"):
        suffix = Path(upload.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.getbuffer())
            tmp_path = tmp.name
        with st.spinner("Running pipeline (render → OCR → CV → VLM)…"):
            try:
                result = demo_core.run_live_extraction(tmp_path)
                st.success("Extraction complete.")
                _render_fields(result)
            except Exception as e:  # noqa: BLE001
                st.error(f"Extraction failed: {e}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)
