"""
=====================================================================
FULL PIPELINE — PDF -> local LLM extraction/evaluation -> figures
=====================================================================
Run as a script:      python full_pipeline.py
Run inside Jupyter:   %run full_pipeline.py
Import for dashboard: from full_pipeline import ...  (does NOT auto-run
                       the pipeline; call main() or run this file directly)

WHAT THIS SCRIPT DOES (plain English, matches the assignment brief)
---------------------------------------------------------------------
1. Reads the assigned PDF report and splits it into chunks + chapters.
2. Uses one LLM (EXTRACTOR_MODEL) to pull out: key results, chapter
   summaries, theme counts, strengths/challenges, numerical indicators,
   and time-series trends.
3. Uses a DIFFERENT LLM (EVALUATOR_MODEL) to mark the extractor's work
   for consistency, completeness and factual alignment with the source.
4. Runs a 3-model comparison (extra credit), including one vision model
   (qwen2.5vl:7b) that reads page IMAGES directly for numbers that plain
   text extraction misses (charts, infographics, scanned pages).
5. Builds a set of clean Plotly charts for the dashboard/report.

Design rules followed throughout:
- Extraction always scans the FULL document (every chunk), not just the
  first page or two, so nothing important later in the report is missed.
- A chart NEVER draws a missing value as 0 or "N/A" pretending it's real —
  missing values are simply left out of that chart rather than faked.
- Charts that would otherwise mix very different units (e.g. HDI ~0-1 vs
  population in millions) are split into small multiples instead of one
  crowded shared axis.

Outputs:
  outputs/*.json   -> machine-readable extraction + evaluation results
  figures/*.png    -> static images for the PDF report
  figures/*.html   -> interactive copies (optional, kept for convenience)
=====================================================================
"""

import os
import re
import io
import json
import time
import base64
from collections import Counter

import ollama
import pdfplumber
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

# =====================================================================
# 1. CONFIG — edit this block for your assignment
# =====================================================================

EXTRACTOR_MODEL = "llama3.2:3b"          # extracts + summarises (text-only)
EVALUATOR_MODEL = "qwen2.5:3b"           # judges the extractor's output

# Three models for the cross-LLM comparison (extra credit). One is a
# vision-capable model (qwen2.5vl) so the comparison covers text+image
# extraction as well as pure text. Pull them first:
#   ollama pull llama3.2:3b
#   ollama pull qwen2.5:3b
#   ollama pull qwen2.5vl:7b
COMPARISON_MODELS = ["llama3.2:3b", "qwen2.5:3b", "qwen2.5vl:7b"]

# Vision (multimodal) model used to read charts/infographics/scanned pages
# directly as images, merged into the main indicator extraction.
VISION_MODEL = "qwen2.5vl:7b"
RUN_VISION_EXTRACTION = True     # set False to skip vision pass entirely
MAX_VISION_PAGES = 15            # cap pages sent to the vision model (cost/time control)
VISION_IMAGE_RESOLUTION = 120    # DPI for rendered page images sent to the vision model

PDF_PATH = r"Montenegro_National-Human-Development-Report-2016_Informal-Work.pdf"

# Bigger chunks = fewer LLM calls = much faster (800 -> 1500 roughly halves total calls)
CHUNK_SIZE_WORDS = 1500
CHUNK_OVERLAP_WORDS = 100

RUN_COMPARISON = True   # set False to skip the extra-credit 3-model comparison

NUM_PREDICT_SHORT = 220   # theme counts, indicators, strengths/challenges, trends, evaluation
NUM_PREDICT_LONG = 400    # chapter summaries, key results
KEEP_ALIVE = "30m"

# PNG export is the primary deliverable for the report — figures/*.png must
# exist and be viewable. HTML copies are also written for interactivity.
EXPORT_PNG = True
ALSO_SAVE_HTML = True

# The 7 themes required by the assignment brief, plus "informality" — this
# specific report (Montenegro, Informal Work) is organised almost entirely
# around the informal/grey economy, so giving it its own theme (instead of
# folding it into "economy"/"employment") gives a much more honest chart.
THEMES = ["education", "health", "inequality", "economy", "gender", "climate",
          "employment", "informality"]

# Numerical indicator schema. First 7 are the generic HDI-report fields the
# assignment brief gives as an EXAMPLE. The next 6 are specific to THIS
# report (pulled from its own contents page) since it's built around the
# informal economy, not a generic HDI writeup.
INDICATOR_KEYS = [
    "hdi_value", "hdi_rank", "life_expectancy_years", "expected_years_schooling",
    "mean_years_schooling", "gni_per_capita_usd", "population",
    "informal_employment_rate_pct", "real_gdp_growth_rate_pct",
    "poverty_risk_rate_pct", "epl_index", "female_activity_rate_pct",
    "male_activity_rate_pct",
]

INDICATOR_LABELS = {
    "hdi_value": "HDI value", "hdi_rank": "HDI rank",
    "life_expectancy_years": "Life expectancy (years)",
    "expected_years_schooling": "Expected years of schooling",
    "mean_years_schooling": "Mean years of schooling",
    "gni_per_capita_usd": "GNI per capita (US$)", "population": "Population",
    "informal_employment_rate_pct": "Informal employment rate (%)",
    "real_gdp_growth_rate_pct": "Real GDP growth rate (%)",
    "poverty_risk_rate_pct": "Poverty risk rate (%)", "epl_index": "EPL index",
    "female_activity_rate_pct": "Female activity rate (%)",
    "male_activity_rate_pct": "Male activity rate (%)",
}

# Headline KPIs shown as gauges — the report's own "at a glance" numbers.
GAUGE_INDICATOR_KEYS = ["informal_employment_rate_pct", "poverty_risk_rate_pct",
                         "real_gdp_growth_rate_pct", "epl_index"]
# Categories compared on the radar chart (all roughly "human development" style).
RADAR_INDICATOR_KEYS = ["hdi_value", "life_expectancy_years",
                         "expected_years_schooling", "mean_years_schooling"]

OUTPUT_DIR = "outputs"
FIGURES_DIR = "figures"

SUMMARIES_PATH = f"{OUTPUT_DIR}/summaries.json"
INDICATORS_PATH = f"{OUTPUT_DIR}/indicators.json"
INDICATORS_TEXT_PATH = f"{OUTPUT_DIR}/indicators_text_only.json"
INDICATORS_VISION_PATH = f"{OUTPUT_DIR}/indicators_vision_only.json"
THEMES_PATH = f"{OUTPUT_DIR}/themes.json"
EVALUATION_PATH = f"{OUTPUT_DIR}/evaluation.json"
TRENDS_PATH = f"{OUTPUT_DIR}/trends.json"
COMPARISON_PATH = f"{OUTPUT_DIR}/model_comparison.json"
KEY_RESULTS_PATH = f"{OUTPUT_DIR}/key_results.json"
STRENGTHS_CHALLENGES_PATH = f"{OUTPUT_DIR}/strengths_challenges.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)


# =====================================================================
# 2. UTILS — Ollama calling, JSON parsing, chunking, cleaning
# =====================================================================

# Sends one text prompt to a local Ollama model and returns its reply as a string.
def call_ollama(model: str, prompt: str, system: str = None, json_mode: bool = False,
                 num_predict: int = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options = {"temperature": 0.2}
    if num_predict:
        options["num_predict"] = num_predict

    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model, messages=messages, options=options,
                format="json" if json_mode else None,
                keep_alive=KEEP_ALIVE,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[call_ollama] attempt {attempt + 1} failed for model={model}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Ollama call failed twice for model {model}")


# Same as call_ollama but also sends one page image, for the vision model (qwen2.5vl).
def call_ollama_vision(model: str, prompt: str, image_b64: str, num_predict: int = None) -> str:
    options = {"temperature": 0.2}
    if num_predict:
        options["num_predict"] = num_predict
    for attempt in range(2):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
                options=options, format="json", keep_alive=KEEP_ALIVE,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"[call_ollama_vision] attempt {attempt + 1} failed for model={model}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Ollama vision call failed twice for model {model}")


# Quick name check: is this Ollama model a multimodal/vision model (e.g. qwen2.5vl, llava)?
def is_vision_model(model_name: str) -> bool:
    name = model_name.lower()
    tags = ["llava", "vision", "moondream", "bakllava",
            "qwen2.5vl", "qwen2-vl", "qwenvl", "-vl", "minicpm-v", "pixtral"]
    return any(tag in name for tag in tags)


# Turns a raw LLM text reply into real JSON, cleaning up common mistakes
# (```json fences, stray prose before/after the JSON). Falls back to
# {"raw_text": ...} if nothing parseable is found.
def safe_json_parse(text: str):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        print("[safe_json_parse] Could not parse JSON, returning raw text wrapped.")
        return {"raw_text": text}


# Forces any parsed indicator output (dict, list, or garbage) into the fixed
# INDICATOR_KEYS schema, so every downstream chart can rely on the same shape.
def normalize_indicators(parsed) -> dict:
    result = {k: None for k in INDICATOR_KEYS}

    if isinstance(parsed, dict):
        if len(parsed) == 1 and isinstance(next(iter(parsed.values())), dict):
            parsed = next(iter(parsed.values()))
        for k in INDICATOR_KEYS:
            if k in parsed:
                result[k] = parsed[k]
        return result

    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return normalize_indicators(parsed[0])
        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip().lower().replace(" ", "_")
                if name in result:
                    result[name] = item.get("value")
        return result

    return result


# Fills any missing (null) value in `primary` (text extraction) using the
# matching value from `fallback` (vision extraction), field by field.
def merge_indicators(primary: dict, fallback: dict) -> dict:
    primary = normalize_indicators(primary)
    fallback = normalize_indicators(fallback)
    merged = dict(primary)
    for k in INDICATOR_KEYS:
        if merged.get(k) is None and fallback.get(k) is not None:
            merged[k] = fallback[k]
    return merged


# Forces any parsed strengths/challenges output into {"strengths": [...], "challenges": [...]}.
def normalize_strengths_challenges(parsed) -> dict:
    result = {"strengths": [], "challenges": []}
    if isinstance(parsed, dict):
        s = parsed.get("strengths", [])
        c = parsed.get("challenges", [])
        result["strengths"] = s if isinstance(s, list) else []
        result["challenges"] = c if isinstance(c, list) else []
    elif isinstance(parsed, list):
        result["strengths"] = parsed
    return result


# Removes duplicate strings (case-insensitive) while keeping the first-seen wording.
def dedupe_strings(items) -> list:
    seen, out = set(), []
    for it in items:
        s = str(it).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {path}")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Splits the full document into overlapping word-count chunks, so each LLM
# call only has to read a manageable slice of text instead of the whole PDF.
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(words):
            break
    return chunks


# Strips page numbers, repeated blank lines, and extra spaces from extracted text.
def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


# =====================================================================
# 3. PDF PROCESSING (Task 1)
# =====================================================================

CHAPTER_PATTERNS = [
    r"^chapter\s+\d+[:\.]?\s*.*$",
    r"^part\s+[ivx\d]+[:\.]?\s*.*$",
    r"^section\s+\d+[:\.]?\s*.*$",
    r"^\d+\.\s+[A-Z][A-Za-z\s]{4,60}$",
]


# Pulls the plain text layer out of every PDF page (charts/scanned pages
# aren't readable this way — see render_page_images for those).
def extract_raw_text(pdf_path: str) -> list:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


# Converts up to `max_pages` PDF pages into base64 PNG images, so the vision
# model (qwen2.5vl) can literally look at charts/infographics/scanned pages.
def render_page_images(pdf_path: str, max_pages: int = MAX_VISION_PAGES,
                        resolution: int = VISION_IMAGE_RESOLUTION) -> list:
    images_b64 = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                pil_image = page.to_image(resolution=resolution).original
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    except Exception as e:
        print(f"[render_page_images] Could not rasterise PDF pages: {e}")
        print("[render_page_images] Skipping vision extraction. Fix: pip install -U pdfplumber pypdfium2 pillow")
    return images_b64


# Splits page text into chapters by looking for heading-style lines
# ("Chapter 1", "1. Introduction", etc.) using the CHAPTER_PATTERNS above.
def detect_chapters(pages: list) -> list:
    chapters = []
    current_title = "Introduction / Front Matter"
    current_text = []
    current_start_page = 1
    combined_patterns = re.compile("|".join(CHAPTER_PATTERNS), re.IGNORECASE)

    for page in pages:
        for line in page["text"].split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if combined_patterns.match(stripped) and len(stripped) < 100:
                if current_text:
                    chapters.append({"title": current_title, "start_page": current_start_page,
                                      "text": clean_text(" ".join(current_text))})
                current_title = stripped
                current_start_page = page["page"]
                current_text = []
            else:
                current_text.append(stripped)

    if current_text:
        chapters.append({"title": current_title, "start_page": current_start_page,
                          "text": clean_text(" ".join(current_text))})
    return chapters


# Alternative to detect_chapters(): use this if automatic detection finds
# only 1 section — pass page ranges you've picked yourself, e.g.
# {"Chapter 1": (29, 45), "Chapter 2": (46, 63)}.
def split_by_manual_ranges(pages: list, ranges: dict) -> list:
    chapters = []
    for title, (start, end) in ranges.items():
        text = " ".join(p["text"] for p in pages if start <= p["page"] <= end)
        chapters.append({"title": title, "start_page": start, "text": clean_text(text)})
    return chapters


# Top-level Task 1 step: load the PDF, get full text + chapters + chunks ready for extraction.
def process_pdf(pdf_path: str = PDF_PATH, manual_ranges: dict = None):
    print(f"Loading PDF: {pdf_path}")
    pages = extract_raw_text(pdf_path)
    full_text = clean_text(" ".join(p["text"] for p in pages))
    print(f"Extracted {len(full_text.split())} words across {len(pages)} pages")

    if manual_ranges:
        chapters = split_by_manual_ranges(pages, manual_ranges)
    else:
        chapters = detect_chapters(pages)
        if len(chapters) <= 1:
            print("WARNING: only 1 section detected — consider passing manual_ranges=")

    chunks = chunk_text(full_text)
    print(f"Split into {len(chunks)} overlapping chunks for LLM processing")
    return full_text, chapters, chunks


# =====================================================================
# 4. PROMPTS — quote these directly in your report
# =====================================================================

THEMES_LIST = ", ".join(THEMES)

KEY_RESULTS_PROMPT = """You are an analyst summarising a national human development report focused on the
informal economy. Read the text below and produce up to 8 bullet points capturing the most
important findings visible IN THIS EXCERPT (e.g. HDI trend, scale of informal employment,
gender gaps, poverty risk, policy findings).
If this excerpt has few or no notable findings, return fewer bullets (even an empty list).

Rules:
- Only use facts present in the text. Do not invent numbers.
- Each bullet must be a single, concise sentence.
- Return ONLY a JSON list of strings, nothing else.

TEXT:
{text}
"""

KEY_RESULTS_CONSOLIDATE_PROMPT = """Below are candidate key-result bullet points gathered from
every section of the report. Select and lightly merge them into the 5 to 8 most important,
non-redundant bullet points overall.

Return ONLY a JSON list of strings, nothing else.

CANDIDATES:
{candidates}
"""

CHAPTER_SUMMARY_PROMPT = """You are summarising one chapter of a national development report.

Chapter title: {title}

Summarise this chapter in under 100 words. Focus on the chapter's main argument,
key statistics mentioned, and its conclusion. Do not add information not present
in the text below.

Return ONLY the summary text, no preamble, no headings.

CHAPTER TEXT:
{text}
"""

THEME_COUNT_PROMPT = f"""You are classifying a passage from a development report into themes.
The possible themes are: {THEMES_LIST}.

Read the passage and estimate how many distinct sentences/ideas relate to each theme.
A passage can relate to multiple themes.

Return ONLY valid JSON in exactly this format (integers, no explanation):
{{{{"education": 0, "health": 0, "inequality": 0, "economy": 0, "gender": 0, "climate": 0, "employment": 0, "informality": 0}}}}

PASSAGE:
{{text}}
"""

STRENGTHS_CHALLENGES_PROMPT = """You are extracting key strengths and challenges from a country development report.

Read the text below (a section of a larger report) and extract any strengths and challenges
explicitly supported by THIS EXCERPT only:
- "strengths": short items (e.g. "high life expectancy", "strong primary education enrolment")
- "challenges": short items (e.g. "income inequality", "gender gap in employment")

Only use what is explicitly supported by the text. Keep each item under 10 words.
If this excerpt has no clear strengths or challenges, return empty lists.

Return ONLY valid JSON in this format:
{"strengths": ["...", "..."], "challenges": ["...", "..."]}

TEXT:
<<<TEXT>>>
"""

STRENGTHS_CHALLENGES_CONSOLIDATE_PROMPT = """Below are candidate '{label}' items gathered from
every section of a country development report. Select and lightly merge them into the
5 to 8 best, non-redundant items. Keep each item under 10 words.

Return ONLY a JSON list of strings, nothing else.

CANDIDATES:
{candidates}
"""

INDICATORS_PROMPT = """You are extracting structured numerical indicators from a national development report
about Montenegro's informal economy.

Find and extract these indicators if present in the text below (use null if not found
IN THIS EXCERPT):
- HDI value
- HDI rank
- Life expectancy at birth (years)
- Expected years of schooling
- Mean years of schooling
- GNI per capita (US$)
- Population (total)
- Informal employment rate (% of total employment)
- Real GDP growth rate (%)
- Poverty risk rate (%)
- EPL index (Employment Protection Legislation index)
- Female activity rate, age 15-64 (%)
- Male activity rate, age 15-64 (%)

Only extract numbers explicitly stated in the text. Do not estimate or calculate.

Return ONLY valid JSON in this exact format (a single JSON object, not a list):
{
  "hdi_value": null, "hdi_rank": null, "life_expectancy_years": null,
  "expected_years_schooling": null, "mean_years_schooling": null,
  "gni_per_capita_usd": null, "population": null,
  "informal_employment_rate_pct": null, "real_gdp_growth_rate_pct": null,
  "poverty_risk_rate_pct": null, "epl_index": null,
  "female_activity_rate_pct": null, "male_activity_rate_pct": null
}

TEXT:
<<<TEXT>>>
"""

# Same schema, written for the vision model (qwen2.5vl) reading a page IMAGE.
VISION_INDICATORS_PROMPT = """Look at this page image from a national development report about
Montenegro's informal economy. It may contain charts, infographics, tables, or
scanned text that plain text extraction would miss.

If you can see any of these indicators anywhere on this page (in body text, a
chart, an infographic panel, or a table), extract their values. Use null for
anything not visible on this page. Do not guess or estimate — only report a
value if it is actually shown on the page.

- HDI value, HDI rank, Life expectancy (years), Expected/Mean years of schooling
- GNI per capita (US$), Population (total)
- Informal employment rate (%), Real GDP growth rate (%), Poverty risk rate (%)
- EPL index, Female activity rate (%), Male activity rate (%)

Return ONLY valid JSON in this exact format:
{
  "hdi_value": null, "hdi_rank": null, "life_expectancy_years": null,
  "expected_years_schooling": null, "mean_years_schooling": null,
  "gni_per_capita_usd": null, "population": null,
  "informal_employment_rate_pct": null, "real_gdp_growth_rate_pct": null,
  "poverty_risk_rate_pct": null, "epl_index": null,
  "female_activity_rate_pct": null, "male_activity_rate_pct": null
}
"""

DEMOGRAPHIC_TRENDS_PROMPT = """You are extracting time-series data from a development report for plotting.

Look for any values reported across multiple years (e.g. HDI over time, population growth,
life expectancy over time, informal employment over time). Extract every (year, value) pair
you find, and use a short, consistent metric name (e.g. "HDI", "Population growth",
"Informal employment rate").

Return ONLY valid JSON in this format:
{
  "series": [
    {"metric": "HDI", "unit": "index (0-1)", "data": [{"year": 2010, "value": 0.0}]}
  ]
}

If no time-series data is present, return {"series": []}.

TEXT:
<<<TEXT>>>
"""

EVALUATE_SUMMARY_PROMPT = """You are a strict quality reviewer. You are given a SOURCE TEXT and a SUMMARY
that another AI model produced from it. Score the summary on three criteria from 1 (poor) to 5 (excellent):
- "consistency": does the summary logically match the source without contradiction?
- "completeness": does it cover the main points of the source?
- "factual_alignment": are all facts/numbers in the summary actually present in the source?

Return ONLY valid JSON:
{"consistency": 0, "completeness": 0, "factual_alignment": 0, "notes": "one sentence justification"}

SOURCE TEXT:
<<<SOURCE>>>

SUMMARY:
<<<SUMMARY>>>
"""

EVALUATE_INDICATORS_PROMPT = """You are a strict fact-checker. You are given SOURCE TEXT and a JSON of
extracted numerical INDICATORS. Check whether each non-null value actually appears in the source text.

Return ONLY valid JSON:
{"correct_fields": ["..."], "incorrect_or_unsupported_fields": ["..."], "accuracy_score_1to5": 0}

SOURCE TEXT:
<<<SOURCE>>>

EXTRACTED INDICATORS:
<<<INDICATORS>>>
"""


# =====================================================================
# 5. EXTRACTION (Task 1 + Task 2) — uses EXTRACTOR_MODEL (+ VISION_MODEL)
# =====================================================================

# Scans EVERY chunk of the full report (not just the intro) for key results,
# pools all the candidate bullets, removes duplicates, then asks the model to
# pick the best 5-8 overall so later chapters get a fair chance to contribute.
def extract_key_results(full_text: str, model: str = EXTRACTOR_MODEL) -> list:
    chunks = chunk_text(full_text)
    candidates = []
    for i, chunk in enumerate(chunks):
        raw = call_ollama(model, KEY_RESULTS_PROMPT.format(text=chunk), json_mode=True,
                           num_predict=NUM_PREDICT_LONG)
        result = safe_json_parse(raw)
        if isinstance(result, list):
            candidates.extend(str(x) for x in result)
        elif isinstance(result, dict):
            values = result.get("raw_text", list(result.values()))
            candidates.extend(str(x) for x in (values if isinstance(values, list) else [values]))
        print(f"Key-results scan {i + 1}/{len(chunks)} done")

    deduped = dedupe_strings(candidates)
    if not deduped:
        return []
    if len(deduped) <= 8:
        return deduped

    pool_text = "\n".join(f"- {b}" for b in deduped)
    raw = call_ollama(model, KEY_RESULTS_CONSOLIDATE_PROMPT.format(candidates=pool_text),
                       json_mode=True, num_predict=NUM_PREDICT_LONG)
    result = safe_json_parse(raw)
    return result if isinstance(result, list) and result else deduped[:8]


# Summarises each detected chapter in under 100 words (skips tiny fragments).
def summarise_chapters(chapters: list, model: str = EXTRACTOR_MODEL) -> list:
    summaries = []
    for chapter in chapters:
        text = chapter["text"]
        if len(text.split()) < 20:
            continue
        excerpt = " ".join(text.split()[:2500])
        prompt = CHAPTER_SUMMARY_PROMPT.format(title=chapter["title"], text=excerpt)
        summary = call_ollama(model, prompt, num_predict=NUM_PREDICT_LONG)
        summaries.append({"title": chapter["title"], "start_page": chapter["start_page"], "summary": summary})
        print(f"Summarised chapter: {chapter['title']}")
    return summaries


# Counts how many sentences/ideas in each chunk relate to each theme, then adds it all up.
def extract_theme_counts(chunks: list, model: str = EXTRACTOR_MODEL) -> dict:
    total_counts = Counter({t: 0 for t in THEMES})
    for i, chunk in enumerate(chunks):
        raw = call_ollama(model, THEME_COUNT_PROMPT.format(text=chunk), json_mode=True,
                           num_predict=NUM_PREDICT_SHORT)
        parsed = safe_json_parse(raw)
        if isinstance(parsed, dict):
            for theme in THEMES:
                total_counts[theme] += int(parsed.get(theme, 0) or 0)
        print(f"Theme pass {i + 1}/{len(chunks)} done")
    return dict(total_counts)


# Same full-document, pool-then-consolidate approach as extract_key_results,
# but for strengths and challenges.
def extract_strengths_challenges(full_text: str, model: str = EXTRACTOR_MODEL) -> dict:
    chunks = chunk_text(full_text)
    all_strengths, all_challenges = [], []
    for i, chunk in enumerate(chunks):
        prompt = STRENGTHS_CHALLENGES_PROMPT.replace("<<<TEXT>>>", chunk)
        raw = call_ollama(model, prompt, json_mode=True, num_predict=NUM_PREDICT_SHORT)
        parsed = normalize_strengths_challenges(safe_json_parse(raw))
        all_strengths.extend(parsed["strengths"])
        all_challenges.extend(parsed["challenges"])
        print(f"Strengths/challenges scan {i + 1}/{len(chunks)} done")

    all_strengths = dedupe_strings(all_strengths)
    all_challenges = dedupe_strings(all_challenges)

    def consolidate(items, label):
        if len(items) <= 8:
            return items
        pool_text = "\n".join(f"- {b}" for b in items)
        raw = call_ollama(model, STRENGTHS_CHALLENGES_CONSOLIDATE_PROMPT.format(label=label, candidates=pool_text),
                           json_mode=True, num_predict=NUM_PREDICT_SHORT)
        result = safe_json_parse(raw)
        return result if isinstance(result, list) and result else items[:8]

    return {"strengths": consolidate(all_strengths, "strengths"),
            "challenges": consolidate(all_challenges, "challenges")}


# Scans every chunk of the full report for numerical indicators, keeping the
# first real (non-null) value found for each field. Stops early once every
# field in the schema has been filled in.
def extract_indicators(full_text: str, model: str = EXTRACTOR_MODEL) -> dict:
    chunks = chunk_text(full_text)
    merged = {k: None for k in INDICATOR_KEYS}
    for i, chunk in enumerate(chunks):
        prompt = INDICATORS_PROMPT.replace("<<<TEXT>>>", chunk)
        raw = call_ollama(model, prompt, json_mode=True, num_predict=NUM_PREDICT_SHORT)
        parsed = normalize_indicators(safe_json_parse(raw))
        for k in INDICATOR_KEYS:
            if merged[k] is None and parsed.get(k) is not None:
                merged[k] = parsed[k]
        print(f"Indicator scan {i + 1}/{len(chunks)} done")
        if all(v is not None for v in merged.values()):
            print("Indicator scan: all fields found, stopping early.")
            break
    return merged


# Same idea as extract_indicators, but reads rendered PAGE IMAGES with the
# vision model (qwen2.5vl) — catches numbers hidden in charts/tables/scans.
def extract_indicators_vision(pdf_path: str, model: str = VISION_MODEL,
                               max_pages: int = MAX_VISION_PAGES) -> dict:
    images = render_page_images(pdf_path, max_pages=max_pages)
    if not images:
        return {k: None for k in INDICATOR_KEYS}

    merged = {k: None for k in INDICATOR_KEYS}
    for i, image_b64 in enumerate(images):
        try:
            raw = call_ollama_vision(model, VISION_INDICATORS_PROMPT, image_b64, num_predict=NUM_PREDICT_SHORT)
            parsed = normalize_indicators(safe_json_parse(raw))
        except Exception as e:
            print(f"[extract_indicators_vision] page {i + 1} failed: {e}")
            continue
        for k in INDICATOR_KEYS:
            if merged[k] is None and parsed.get(k) is not None:
                merged[k] = parsed[k]
        print(f"Vision scan page {i + 1}/{len(images)} done")
        if all(v is not None for v in merged.values()):
            print("Vision scan: all indicator fields found, stopping early.")
            break
    return merged


# Pulls out every (year, value) series mentioned anywhere in the report,
# merges duplicate metric names (case-insensitive), and keeps only the
# richest 6 series so the trend chart stays readable.
def extract_demographic_trends(chunks: list, model: str = EXTRACTOR_MODEL, max_series: int = 6) -> dict:
    all_series = {}          # canonical_lowercase_name -> {"label":..., "unit":..., "data": {year: value}}
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")

    relevant_chunks = [c for c in chunks if year_pattern.search(c)]
    skipped = len(chunks) - len(relevant_chunks)
    if skipped:
        print(f"Skipping {skipped}/{len(chunks)} chunks with no year mentioned (trend scan)")

    for i, chunk in enumerate(relevant_chunks):
        prompt = DEMOGRAPHIC_TRENDS_PROMPT.replace("<<<TEXT>>>", chunk)
        raw = call_ollama(model, prompt, json_mode=True, num_predict=NUM_PREDICT_SHORT)
        parsed = safe_json_parse(raw)
        series_list = parsed.get("series", []) if isinstance(parsed, dict) else []
        for s in series_list:
            if not isinstance(s, dict):
                continue
            label = str(s.get("metric", "unknown")).strip()
            key = label.lower()
            if key not in all_series:
                all_series[key] = {"label": label, "unit": s.get("unit", ""), "data": {}}
            for point in s.get("data", []):
                if not isinstance(point, dict):
                    continue
                year, value = point.get("year"), point.get("value")
                # only keep genuinely present values — never fake a missing one as 0
                if year is not None and value is not None:
                    all_series[key]["data"][year] = value
        print(f"Trend scan {i + 1}/{len(relevant_chunks)} done")

    # keep only the series with the most actual data points, capped at max_series,
    # so the chart doesn't get cluttered with one-off duplicate/near-empty metrics
    ranked = sorted(all_series.values(), key=lambda s: len(s["data"]), reverse=True)[:max_series]
    series_out = [{"metric": s["label"], "unit": s["unit"],
                   "data": [{"year": y, "value": v} for y, v in sorted(s["data"].items())]}
                  for s in ranked]
    return {"series": series_out}


# =====================================================================
# 6. EVALUATION — uses EVALUATOR_MODEL (a different LLM judges Task 5's output)
# =====================================================================

# Asks the evaluator model to score each chapter summary against its source text.
def evaluate_chapter_summaries(full_text: str, chapter_summaries: list, model: str = EVALUATOR_MODEL) -> list:
    results = []
    source_excerpt = " ".join(full_text.split()[:2500])
    for item in chapter_summaries:
        prompt = EVALUATE_SUMMARY_PROMPT.replace("<<<SOURCE>>>", source_excerpt).replace("<<<SUMMARY>>>", item["summary"])
        raw = call_ollama(model, prompt, json_mode=True, num_predict=NUM_PREDICT_SHORT)
        scores = safe_json_parse(raw)
        if not isinstance(scores, dict):
            scores = {"raw_text": raw}
        results.append({"title": item["title"], **scores})
        print(f"Evaluated summary: {item['title']}")
    return results


# Asks the evaluator model to fact-check the extracted indicators against the source text,
# returning a 1-5 accuracy score. Used both for the main run and each comparison model.
def evaluate_indicators(full_text: str, indicators: dict, model: str = EVALUATOR_MODEL) -> dict:
    source_excerpt = " ".join(full_text.split()[:3500])
    prompt = EVALUATE_INDICATORS_PROMPT.replace("<<<SOURCE>>>", source_excerpt).replace(
        "<<<INDICATORS>>>", json.dumps(indicators))
    raw = call_ollama(model, prompt, json_mode=True, num_predict=NUM_PREDICT_SHORT)
    parsed = safe_json_parse(raw)
    return parsed if isinstance(parsed, dict) else {"raw_text": raw}


# =====================================================================
# 7. VISUALISATION (Task 3) — Plotly figures
# Every chart here follows two rules: (1) never draw a missing value as a
# fake zero — just leave it out, and (2) never mix wildly different units
# on one shared axis — use small multiples instead.
# =====================================================================

# Shared helper: builds a small-multiples bar chart (one clean mini-chart
# per category), instead of cramming everything onto a single axis.
def _small_multiples_bar(df: pd.DataFrame, x: str, y: str, facet: str, color: str,
                          title: str, value_fmt: str = "%{text:.2f}", wrap: int = 4):
    n = df[facet].nunique()
    fig = px.bar(df, x=x, y=y, color=color, text=y, facet_col=facet,
                 facet_col_wrap=min(wrap, n), title=title)
    fig.update_traces(texttemplate=value_fmt, textposition="outside")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(height=320 + 220 * ((n - 1) // wrap))
    return fig


# Bar chart of how many times each theme came up across the whole report.
def plot_theme_distribution(theme_counts: dict):
    theme_counts = theme_counts if isinstance(theme_counts, dict) else {}
    df = pd.DataFrame(list(theme_counts.items()), columns=["Theme", "Count"]).sort_values("Count", ascending=False)
    fig = px.bar(df, x="Theme", y="Count", title="Distribution of Themes in the Report",
                 color="Theme", text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


# Compares extracted indicators across models. Each indicator gets its own
# mini-chart with its own y-axis, since HDI value (~0-1), life expectancy
# (~75) and population (millions) can't share one axis without one of them
# becoming invisible. Only real (non-null) values are ever plotted.
def plot_indicator_comparison(indicators_by_model: dict):
    rows = []
    for model_name, indicators in indicators_by_model.items():
        indicators = normalize_indicators(indicators)
        for key, value in indicators.items():
            if isinstance(value, (int, float)):
                rows.append({"Model": model_name, "Indicator": INDICATOR_LABELS.get(key, key), "Value": value})
    if not rows:
        return go.Figure().update_layout(title="No numerical indicators available to compare")
    df = pd.DataFrame(rows)
    # keep the chart readable: show at most the 8 indicators with the most data
    top_indicators = df["Indicator"].value_counts().head(8).index
    df = df[df["Indicator"].isin(top_indicators)]
    return _small_multiples_bar(df, x="Model", y="Value", facet="Indicator", color="Model",
                                 title="Model Comparison of Extracted Numerical Indicators")


# One clean mini line-chart per time-series metric (HDI over time, population
# growth, etc.), each with its own y-axis, so nothing gets squashed flat by
# a metric on a totally different scale. Only the endpoint of each line is
# labelled with its value, to avoid overlapping text along the whole line.
def plot_demographic_trend(trends: dict):
    series_list = trends.get("series", []) if isinstance(trends, dict) else []
    series_list = [s for s in series_list if isinstance(s, dict) and s.get("data")]
    if not series_list:
        return go.Figure().update_layout(title="No time-series data found in this report")

    n = len(series_list)
    cols = min(3, n)
    rows = (n - 1) // cols + 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[s.get("metric", "series") for s in series_list])

    for i, s in enumerate(series_list):
        r, c = i // cols + 1, i % cols + 1
        data = [d for d in s["data"] if isinstance(d, dict) and "year" in d and "value" in d]
        years = [d["year"] for d in data]
        values = [d["value"] for d in data]
        fig.add_trace(
            go.Scatter(x=years, y=values, mode="lines+markers", showlegend=False),
            row=r, col=c,
        )
        if values:
            fig.add_annotation(x=years[-1], y=values[-1], text=f"{values[-1]}", showarrow=False,
                                yshift=12, row=r, col=c, font=dict(size=11))

    fig.update_layout(title="Demographic / Development Trends Over Time (each metric on its own scale)",
                       height=320 * rows)
    return fig


# Two-sided bar showing strengths (right) vs challenges (left).
def plot_strengths_challenges(strengths_challenges: dict):
    strengths_challenges = normalize_strengths_challenges(strengths_challenges)
    strengths = strengths_challenges.get("strengths", [])
    challenges = strengths_challenges.get("challenges", [])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=strengths, x=[1] * len(strengths), orientation="h", name="Strengths",
                          marker_color="green", text=strengths, textposition="inside"))
    fig.add_trace(go.Bar(y=challenges, x=[-1] * len(challenges), orientation="h", name="Challenges",
                          marker_color="firebrick", text=challenges, textposition="inside"))
    fig.update_layout(title="Key Strengths vs Challenges", barmode="overlay",
                       xaxis=dict(showticklabels=False, zeroline=True), showlegend=True)
    return fig


# Radar chart comparing models on 4 human-development indicators. Values are
# min-max normalised (0-1) per indicator across models, since raw HDI value
# (~0-1), life expectancy (~75), and years of schooling (~12) live on
# completely different scales — plotting them raw would make one axis dwarf
# the rest. Only models with a REAL value for every category are included
# (no faking a missing category as a low score); if fewer than 2 models
# qualify, falls back to a plain small-multiples bar of whatever data exists.
def plot_radar_indicators(indicators_by_model: dict):
    labels = [INDICATOR_LABELS.get(c, c) for c in RADAR_INDICATOR_KEYS]
    raw = {m: normalize_indicators(ind) for m, ind in indicators_by_model.items()}
    complete = {m: [raw[m][c] for c in RADAR_INDICATOR_KEYS] for m in raw
                if all(isinstance(raw[m][c], (int, float)) for c in RADAR_INDICATOR_KEYS)}

    if len(complete) < 2:
        rows = [{"Model": m, "Indicator": INDICATOR_LABELS.get(c, c), "Value": raw[m][c]}
                for m in raw for c in RADAR_INDICATOR_KEYS if isinstance(raw[m][c], (int, float))]
        if not rows:
            return go.Figure().update_layout(title="No complete indicator data available for a model comparison")
        df = pd.DataFrame(rows)
        fig = _small_multiples_bar(df, x="Model", y="Value", facet="Indicator", color="Model",
                                    title="Development Indicators by Model (radar needs ≥2 models with full data)")
        return fig

    normed = {m: [0.0] * len(RADAR_INDICATOR_KEYS) for m in complete}
    for ci in range(len(RADAR_INDICATOR_KEYS)):
        col_vals = [complete[m][ci] for m in complete]
        lo, hi = min(col_vals), max(col_vals)
        for m in complete:
            normed[m][ci] = 0.5 if hi == lo else (complete[m][ci] - lo) / (hi - lo)

    fig = go.Figure()
    for model_name, values in normed.items():
        vals = values + [values[0]]
        hover_actual = complete[model_name] + [complete[model_name][0]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=labels + [labels[0]], fill="toself", name=model_name,
                                       customdata=hover_actual,
                                       hovertemplate="%{theta}: %{customdata}<extra>%{fullData.name}</extra>"))
    fig.update_layout(title="Radar Comparison of Development Indicators (normalised 0-1 per indicator)",
                       polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    return fig


# Compares the 3 models on 3 fair, simple metrics — verbosity (word count),
# thematic richness (theme mentions), and accuracy (evaluator's 1-5 score)
# — each in its own mini-chart. This is the actual "trade-offs between
# accuracy and verbosity" comparison the assignment extra-credit asks for.
def plot_model_stability(comparison_data: dict):
    rows = []
    for model_name, data in comparison_data.items():
        data = data if isinstance(data, dict) else {}
        theme_counts = data.get("theme_counts", {}) or {}
        word_count = data.get("word_count")
        theme_total = sum(v for v in theme_counts.values() if isinstance(v, (int, float))) if theme_counts else None
        accuracy = data.get("accuracy_score")

        if isinstance(word_count, (int, float)):
            rows.append({"Model": model_name, "Metric": "Verbosity (word count)", "Value": word_count})
        if theme_total is not None:
            rows.append({"Model": model_name, "Metric": "Thematic richness (mentions)", "Value": theme_total})
        if isinstance(accuracy, (int, float)):
            rows.append({"Model": model_name, "Metric": "Accuracy (evaluator score, 1-5)", "Value": accuracy})

    if not rows:
        return go.Figure().update_layout(title="No comparison data available")
    df = pd.DataFrame(rows)
    return _small_multiples_bar(df, x="Model", y="Value", facet="Metric", color="Model",
                                 title="Cross-LLM Behaviour: Verbosity vs Thematic Richness vs Accuracy",
                                 value_fmt="%{text:.1f}", wrap=3)


# Bars for how many of the 7 core indicator fields were found by text-only
# extraction vs after merging in the vision model's reading of page images.
def plot_indicator_completeness(indicators_text: dict, indicators_merged: dict):
    text_count = sum(1 for v in normalize_indicators(indicators_text).values() if v is not None)
    merged_count = sum(1 for v in normalize_indicators(indicators_merged).values() if v is not None)
    df = pd.DataFrame({"Extraction method": ["Text-only", "Text + Vision merged"],
                        "Indicators found": [text_count, merged_count]})
    fig = px.bar(df, x="Extraction method", y="Indicators found", color="Extraction method",
                 title="Indicator Completeness: Text-only vs Text+Vision Extraction",
                 range_y=[0, len(INDICATOR_KEYS)], text="Indicators found")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


# Clean table of every extracted indicator value — the one place to see every
# number at a glance. Missing values show as "Not found in report" so it's
# never mistaken for a real 0.
def plot_indicator_summary_table(indicators: dict, title: str = "Extracted Numerical Indicators — Summary"):
    indicators = normalize_indicators(indicators)
    row_labels = [INDICATOR_LABELS.get(k, k) for k in INDICATOR_KEYS]
    row_values = ["Not found in report" if indicators.get(k) is None else indicators.get(k) for k in INDICATOR_KEYS]
    row_colors = ["#f7f7f7" if i % 2 == 0 else "white" for i in range(len(row_labels))]
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Indicator", "Value"], fill_color="#2c3e50",
                    font=dict(color="white", size=13), align="left", height=32),
        cells=dict(values=[row_labels, row_values], fill_color=[row_colors, row_colors],
                   align="left", height=28, font=dict(size=12)),
    )])
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# KPI gauges for this report's own headline numbers. Only indicators that
# were actually found get a gauge — nothing is shown as a fake 0/N/A gauge.
def plot_indicator_gauges(indicators: dict, title: str = "Informal Economy — Headline Indicators"):
    indicators = normalize_indicators(indicators)
    found = [(k, indicators[k]) for k in GAUGE_INDICATOR_KEYS if isinstance(indicators.get(k), (int, float))]
    if not found:
        return go.Figure().update_layout(title=f"{title} — none of these indicators were found in the report")

    fig = make_subplots(rows=1, cols=len(found), specs=[[{"type": "indicator"}] * len(found)])
    for i, (key, value) in enumerate(found):
        label = INDICATOR_LABELS.get(key, key)
        fig.add_trace(go.Indicator(mode="gauge+number", value=value, title={"text": label, "font": {"size": 13}},
                                    number={"suffix": "%" if key.endswith("_pct") else ""}),
                      row=1, col=i + 1)
    fig.update_layout(title=title, margin=dict(t=80, b=10))
    return fig


# Male vs female labour-force activity rate — the report's own gender
# chapter (Fig 1.2.4/1.2.5) is built around this exact comparison. Only
# bars for values actually found are drawn.
def plot_gender_activity_gap(indicators: dict):
    indicators = normalize_indicators(indicators)
    pairs = [("Female", indicators.get("female_activity_rate_pct")),
             ("Male", indicators.get("male_activity_rate_pct"))]
    pairs = [(g, v) for g, v in pairs if isinstance(v, (int, float))]
    if not pairs:
        return go.Figure().update_layout(title="Gender Activity-Rate Gap — no values extracted for this report")

    df = pd.DataFrame(pairs, columns=["Gender", "Activity rate (%)"])
    fig = px.bar(df, x="Gender", y="Activity rate (%)", color="Gender",
                 title="Labour-Force Activity Rate by Gender (age 15-64)", text="Activity rate (%)",
                 color_discrete_map={"Female": "#c9184a", "Male": "#1d3557"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


# Kaleido needs a real Chrome/Chromium install to export PNGs — this
# downloads one once (cached after) and returns whether PNG export will work.
def ensure_kaleido_chrome() -> bool:
    try:
        import kaleido  # noqa: F401
    except ImportError:
        print("[figures] kaleido is not installed — run: pip install -U kaleido")
        return False
    try:
        pio.get_chrome()
        return True
    except Exception as e:
        print(f"[figures] Could not confirm/install Chrome for Kaleido ({e})")
        print("[figures] Fix by running once in your terminal: plotly_get_chrome")
        return False


# Saves every chart as a PNG (for the report) and, optionally, an interactive HTML copy.
def save_all_figures(figures: dict, out_dir: str = FIGURES_DIR, width: int = 1100, height: int = 650, scale: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    chrome_ready = EXPORT_PNG and ensure_kaleido_chrome()
    if EXPORT_PNG and not chrome_ready:
        print("[figures] WARNING: PNG export unavailable. Run: plotly_get_chrome   (then re-run this script)")

    for name, fig in figures.items():
        if chrome_ready:
            try:
                fig.write_image(os.path.join(out_dir, f"{name}.png"), width=width, height=height, scale=scale)
                print(f"Saved -> {out_dir}/{name}.png")
            except Exception as e:
                print(f"PNG export failed for {name} ({e}). Fix: pip install -U kaleido && plotly_get_chrome")
        if ALSO_SAVE_HTML:
            fig.write_html(os.path.join(out_dir, f"{name}.html"))
            print(f"Saved -> {out_dir}/{name}.html")


# =====================================================================
# 8. RUN THE FULL PIPELINE
# =====================================================================

def main(manual_ranges: dict = None):
    print("=" * 60)
    print("STEP 1: PDF Processing")
    print("=" * 60)
    full_text, chapters, chunks = process_pdf(manual_ranges=manual_ranges)
    # If chapter detection finds only 1 section, re-run with e.g.:
    # main(manual_ranges={"Chapter 1": (29, 45), "Chapter 2": (46, 63)})

    print("\n" + "=" * 60)
    print(f"STEP 2: Extraction using {EXTRACTOR_MODEL} (+ {VISION_MODEL} for images/charts)")
    print("=" * 60)
    key_results = extract_key_results(full_text)
    chapter_summaries = summarise_chapters(chapters)
    theme_counts = extract_theme_counts(chunks)
    strengths_challenges = extract_strengths_challenges(full_text)
    trends = extract_demographic_trends(chunks)

    indicators_text = extract_indicators(full_text)
    if RUN_VISION_EXTRACTION:
        print(f"Running vision-based indicator extraction with {VISION_MODEL} "
              f"(reading up to {MAX_VISION_PAGES} page images for charts/infographics)...")
        indicators_vision = extract_indicators_vision(PDF_PATH, VISION_MODEL)
        indicators = merge_indicators(indicators_text, indicators_vision)
        save_json(indicators_text, INDICATORS_TEXT_PATH)
        save_json(indicators_vision, INDICATORS_VISION_PATH)
    else:
        indicators = indicators_text

    save_json(key_results, KEY_RESULTS_PATH)
    save_json(chapter_summaries, SUMMARIES_PATH)
    save_json(theme_counts, THEMES_PATH)
    save_json(strengths_challenges, STRENGTHS_CHALLENGES_PATH)
    save_json(indicators, INDICATORS_PATH)
    save_json(trends, TRENDS_PATH)

    print("\n" + "=" * 60)
    print(f"STEP 3: Evaluation using {EVALUATOR_MODEL}")
    print("=" * 60)
    summary_evaluation = evaluate_chapter_summaries(full_text, chapter_summaries)
    indicator_evaluation = evaluate_indicators(full_text, indicators)
    save_json({"summary_evaluation": summary_evaluation, "indicator_evaluation": indicator_evaluation},
              EVALUATION_PATH)

    print("\n" + "=" * 60)
    print("STEP 4: Cross-LLM comparison (extra credit) — includes one vision-capable model")
    print("=" * 60)
    comparison_data = {}
    if RUN_COMPARISON:
        for model in COMPARISON_MODELS:
            print(f"\nRunning comparison model: {model} (vision-capable: {is_vision_model(model)})")
            try:
                sample_theme_counts = extract_theme_counts(chunks[:3], model=model)  # sample subset for speed
                sample_indicators_text = extract_indicators(full_text, model=model)

                used_vision = False
                if is_vision_model(model):
                    sample_indicators_vision = extract_indicators_vision(PDF_PATH, model)
                    sample_indicators = merge_indicators(sample_indicators_text, sample_indicators_vision)
                    used_vision = True
                else:
                    sample_indicators = sample_indicators_text

                # real accuracy score (not just a proxy) so the "accuracy vs
                # verbosity" comparison chart is based on actual fact-checking
                acc = evaluate_indicators(full_text, sample_indicators, model=EVALUATOR_MODEL)
                accuracy_score = acc.get("accuracy_score_1to5") if isinstance(acc, dict) else None

                comparison_data[model] = {
                    "theme_counts": sample_theme_counts,
                    "indicators": sample_indicators,
                    "word_count": sum(v for v in sample_theme_counts.values() if isinstance(v, (int, float))),
                    "used_vision": used_vision,
                    "accuracy_score": accuracy_score if isinstance(accuracy_score, (int, float)) else None,
                }
            except Exception as e:
                print(f"Model {model} failed (is it pulled in Ollama? `ollama pull {model}`): {e}")
    else:
        print("Skipped (RUN_COMPARISON = False). Set it to True in the config block to enable.")
    save_json(comparison_data, COMPARISON_PATH)

    print("\n" + "=" * 60)
    print("STEP 5: Visualisation — building and saving all figures")
    print("=" * 60)
    indicators_by_model = {EXTRACTOR_MODEL: indicators}
    for model_name, data in comparison_data.items():
        indicators_by_model[model_name] = normalize_indicators(data.get("indicators", {}))

    figures = {
        "theme_distribution": plot_theme_distribution(theme_counts),
        "indicator_comparison": plot_indicator_comparison(indicators_by_model),
        "demographic_trend": plot_demographic_trend(trends),
        "strengths_challenges": plot_strengths_challenges(strengths_challenges),
        "radar_indicators": plot_radar_indicators(indicators_by_model),
        "indicator_summary_table": plot_indicator_summary_table(indicators),
        "indicator_gauges": plot_indicator_gauges(indicators),
        "gender_activity_gap": plot_gender_activity_gap(indicators),
    }
    if comparison_data:
        figures["model_stability"] = plot_model_stability(comparison_data)
    if RUN_VISION_EXTRACTION:
        figures["indicator_completeness"] = plot_indicator_completeness(indicators_text, indicators)

    save_all_figures(figures)

    print("\nPipeline complete.")
    print(f"JSON outputs in: {OUTPUT_DIR}/")
    print(f"Figures (png{' + html' if ALSO_SAVE_HTML else ''}) in: {FIGURES_DIR}/")
    return figures


if __name__ == "__main__":
    main()
