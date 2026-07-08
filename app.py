"""
=====================================================================
STREAMLIT DASHBOARD — Task 3: Data Visualisation and Dashboard
=====================================================================
Run with:
    streamlit run app.py

Reads the JSON files full_pipeline.py produces (outputs/*.json) and
rebuilds every chart live with Plotly, so the dashboard is interactive
(hover, zoom, legend toggling). Run full_pipeline.py at least once first.
=====================================================================
"""

import os
import json

import streamlit as st
import pandas as pd

from full_pipeline import (
    OUTPUT_DIR, FIGURES_DIR,
    KEY_RESULTS_PATH, SUMMARIES_PATH, THEMES_PATH, STRENGTHS_CHALLENGES_PATH,
    INDICATORS_PATH, INDICATORS_TEXT_PATH, INDICATORS_VISION_PATH,
    TRENDS_PATH, EVALUATION_PATH, COMPARISON_PATH,
    EXTRACTOR_MODEL, EVALUATOR_MODEL, VISION_MODEL, COMPARISON_MODELS,
    load_json, normalize_indicators, normalize_strengths_challenges, is_vision_model,
    plot_theme_distribution, plot_indicator_comparison, plot_demographic_trend,
    plot_strengths_challenges, plot_radar_indicators, plot_model_stability,
    plot_indicator_completeness, plot_indicator_summary_table,
    plot_indicator_gauges, plot_gender_activity_gap,
)

st.set_page_config(page_title="Development Report Dashboard", layout="wide")


# Loads a JSON output file, or returns `default` if it's missing/broken —
# so the dashboard never crashes just because one pipeline step wasn't run.
def try_load(path, default):
    if not os.path.exists(path):
        return default
    try:
        return load_json(path)
    except (json.JSONDecodeError, OSError):
        return default


key_results = try_load(KEY_RESULTS_PATH, [])
chapter_summaries = try_load(SUMMARIES_PATH, [])
theme_counts = try_load(THEMES_PATH, {})
strengths_challenges = normalize_strengths_challenges(try_load(STRENGTHS_CHALLENGES_PATH, {}))
indicators = normalize_indicators(try_load(INDICATORS_PATH, {}))
indicators_text_only = normalize_indicators(try_load(INDICATORS_TEXT_PATH, {}))
indicators_vision_only = normalize_indicators(try_load(INDICATORS_VISION_PATH, {}))
trends = try_load(TRENDS_PATH, {"series": []})
evaluation = try_load(EVALUATION_PATH, {})
comparison_data = try_load(COMPARISON_PATH, {})

pipeline_has_run = os.path.exists(OUTPUT_DIR) and os.path.exists(INDICATORS_PATH)
vision_was_used = os.path.exists(INDICATORS_VISION_PATH)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("Report Dashboard")
st.sidebar.markdown(
    f"**Extractor model (text):** `{EXTRACTOR_MODEL}`  \n"
    f"**Vision model (images/charts):** `{VISION_MODEL}`  \n"
    f"**Evaluator model:** `{EVALUATOR_MODEL}`  \n"
    f"**Comparison models:** {', '.join(COMPARISON_MODELS)}"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Reads JSON files from `outputs/`, produced by `full_pipeline.py`. "
    "Re-run the pipeline to refresh the data, then reload this page."
)

if not pipeline_has_run:
    st.warning("No pipeline outputs found yet. Run `python full_pipeline.py` first, then reload this dashboard.")
    st.stop()


# ---------------------------------------------------------------------
# Header — headline KPIs. Missing values show "Not found" rather than a
# misleading blank/zero metric card.
# ---------------------------------------------------------------------
st.title("Montenegro Informal Economy — Analysis Dashboard")

metric_cols = st.columns(4)
metric_defs = [
    ("HDI value", indicators.get("hdi_value"), ""),
    ("Informal employment rate", indicators.get("informal_employment_rate_pct"), "%"),
    ("Poverty risk rate", indicators.get("poverty_risk_rate_pct"), "%"),
    ("Real GDP growth", indicators.get("real_gdp_growth_rate_pct"), "%"),
]
for col, (label, value, suffix) in zip(metric_cols, metric_defs):
    col.metric(label, f"{value}{suffix}" if value is not None else "Not found")

if vision_was_used:
    st.caption(
        f"Indicators above are merged from text extraction ({EXTRACTOR_MODEL}) "
        f"and vision extraction ({VISION_MODEL} reading page images directly) — "
        "see the Indicators tab for a text-only vs merged breakdown."
    )

st.markdown("---")


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_overview, tab_summaries, tab_themes, tab_indicators, tab_trends, tab_models, tab_eval = st.tabs(
    ["Key Results", "Chapter Summaries", "Themes & Strengths/Challenges",
     "Indicators", "Demographic Trends", "Model Comparison", "Evaluation"]
)

# --- Key results ------------------------------------------------------
with tab_overview:
    st.subheader("Key results (Task 1)")
    if key_results:
        for point in key_results:
            st.markdown(f"- {point}")
    else:
        st.info("No key results extracted.")

# --- Chapter summaries --------------------------------------------------
with tab_summaries:
    st.subheader("Chapter-by-chapter summaries (Task 1)")
    if chapter_summaries:
        for chapter in chapter_summaries:
            with st.expander(f"{chapter.get('title', 'Untitled')} (p.{chapter.get('start_page', '?')})"):
                st.write(chapter.get("summary", ""))
    else:
        st.info("No chapter summaries extracted.")

# --- Themes + strengths/challenges -------------------------------------
with tab_themes:
    st.subheader("Theme distribution (Task 2)")
    st.plotly_chart(plot_theme_distribution(theme_counts), use_container_width=True)

    st.subheader("Strengths vs challenges (Task 2)")
    st.plotly_chart(plot_strengths_challenges(strengths_challenges), use_container_width=True)

    col_s, col_c = st.columns(2)
    with col_s:
        st.markdown("**Strengths**")
        for s in strengths_challenges.get("strengths", []):
            st.markdown(f"- {s}")
    with col_c:
        st.markdown("**Challenges**")
        for c in strengths_challenges.get("challenges", []):
            st.markdown(f"- {c}")

# --- Indicators ----------------------------------------------------------
with tab_indicators:
    st.subheader("Extracted numerical indicators (Task 2)")
    st.plotly_chart(plot_indicator_summary_table(indicators), use_container_width=True)

    st.subheader("Headline informal-economy KPIs")
    st.plotly_chart(plot_indicator_gauges(indicators), use_container_width=True)

    st.subheader("Gender activity-rate gap")
    st.plotly_chart(plot_gender_activity_gap(indicators), use_container_width=True)
    st.caption(
        "This report's gender chapter (Fig 1.2.4/1.2.5) centres on the gap "
        "between male and female labour-force activity rates."
    )

    if vision_was_used:
        st.subheader("Text-only vs Text+Vision extraction")
        st.plotly_chart(plot_indicator_completeness(indicators_text_only, indicators), use_container_width=True)
        compare_df = pd.DataFrame({"Text-only": indicators_text_only, "Vision-only": indicators_vision_only,
                                    "Merged (final)": indicators})
        st.dataframe(compare_df, use_container_width=True)
        st.caption(
            "Vision-only shows what qwen2.5vl read directly off rendered page "
            "images. Merged keeps the text value where both agree, and fills "
            "gaps from vision."
        )

    indicators_by_model = {EXTRACTOR_MODEL: indicators}
    for model_name, data in comparison_data.items():
        indicators_by_model[model_name] = normalize_indicators(data.get("indicators", {}))

    if len(indicators_by_model) > 1:
        st.subheader("Model comparison of indicators")
        st.plotly_chart(plot_indicator_comparison(indicators_by_model), use_container_width=True)

        st.subheader("Radar comparison (extra credit)")
        st.plotly_chart(plot_radar_indicators(indicators_by_model), use_container_width=True)
    else:
        st.info("Only one model's indicators are available. Enable RUN_COMPARISON in full_pipeline.py to compare across models.")

# --- Trends ----------------------------------------------------------------
with tab_trends:
    st.subheader("Demographic / development trends over time (Task 2)")
    st.plotly_chart(plot_demographic_trend(trends), use_container_width=True)
    series_list = trends.get("series", [])
    if series_list:
        chosen = st.selectbox("Inspect raw data points for:", [s.get("metric", "series") for s in series_list])
        for s in series_list:
            if s.get("metric") == chosen:
                st.dataframe(pd.DataFrame(s.get("data", [])), use_container_width=True)

# --- Model comparison / stability -----------------------------------------
with tab_models:
    st.subheader("Cross-LLM behaviour analysis (extra credit)")
    if comparison_data:
        st.plotly_chart(plot_model_stability(comparison_data), use_container_width=True)
        st.markdown(
            "**How to read this:** each panel compares the 3 models on one fair "
            "metric — output verbosity, thematic richness, and evaluator-scored "
            "accuracy (1-5) — so you can see the actual trade-off between how "
            "much a model says and how much of it is correct."
        )
        table_rows = []
        for model_name, data in comparison_data.items():
            table_rows.append({
                "Model": model_name,
                "Vision-capable": is_vision_model(model_name),
                "Word count": data.get("word_count", 0),
                "Accuracy score (1-5)": data.get("accuracy_score"),
                **{f"ind_{k}": v for k, v in normalize_indicators(data.get("indicators", {})).items()},
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    else:
        st.info("Model comparison was not run (RUN_COMPARISON = False in full_pipeline.py).")

# --- Evaluation --------------------------------------------------------------
with tab_eval:
    st.subheader(f"Evaluation by {EVALUATOR_MODEL} (Task 1 + Task 2)")
    summary_eval = evaluation.get("summary_evaluation", [])
    indicator_eval = evaluation.get("indicator_evaluation", {})

    if summary_eval:
        st.markdown("**Chapter summary quality scores**")
        st.dataframe(pd.DataFrame(summary_eval), use_container_width=True)
    else:
        st.info("No summary evaluation found.")

    if indicator_eval:
        st.markdown("**Indicator fact-check**")
        st.json(indicator_eval)
    else:
        st.info("No indicator evaluation found.")

st.markdown("---")
st.caption(f"Static PNG copies of every chart above are saved in `{FIGURES_DIR}/` by full_pipeline.py for your report.")
