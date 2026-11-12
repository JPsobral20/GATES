import os
import glob
import csv
import logging
from transformers import pipeline, AutoTokenizer

# ======================================================
# 1. GLOBAL CONSTANTS
# ======================================================

dataset_folder = "./dataset/dataset/"
models_to_process = ["openai", "nebius", "google"]

NER_MODEL = "d4data/biomedical-ner-all"
NER_MAX_TOKENS = 512  # Model token limit

# ======================================================
# 2. MAIN ENTRY POINT
# ======================================================

def run_ner(run_name: str, timestamp: str, base_override: str = None):
    """
    Run NER on both original clinical notes and generated summaries.
    Only processes files that have been summarized (inferred from openai output folder).
    run_name:  e.g. 'it1_control'
    timestamp: e.g. '2026-03-23_11-28-01'
    """

    base_summaries_folder = base_override if base_override else f"./outputs/{run_name}/{timestamp}/"
    ner_folder = os.path.join(base_summaries_folder, "ner")
    os.makedirs(ner_folder, exist_ok=True)

    logging.info("=" * 60)
    logging.info("NER EXTRACTION STARTED")
    logging.info(f"Run: {run_name} | Timestamp: {timestamp}")
    logging.info(f"NER Model: {NER_MODEL}")
    logging.info(f"Saving outputs to: {ner_folder}")
    logging.info("=" * 60)

    # ======================================================
    # 3. LOAD NER PIPELINE
    # ======================================================

    logging.info("Loading NER model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
        ner_pipeline = pipeline(
            "ner",
            model=NER_MODEL,
            tokenizer=tokenizer,
            aggregation_strategy="first",  # "first" handles subword tokens correctly
            device=-1  # CPU; set to 0 for GPU
        )
        logging.info("NER model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load NER model: {e}")
        return

    # ======================================================
    # 4. HELPER: Chunk text to fit model token limit
    # ======================================================

    def chunk_text(text: str, max_tokens: int = NER_MAX_TOKENS) -> list:
        """Split text into chunks that fit within the model's token limit."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_len = 0

        for word in words:
            word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
            if current_len + word_tokens > max_tokens - 2:  # Reserve for special tokens
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_tokens
            else:
                current_chunk.append(word)
                current_len += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ======================================================
    # 5. HELPER: Run NER on a full text (with chunking)
    # ======================================================

    def extract_entities(text: str) -> list:
        """Extract all entities from text, handling long documents via chunking."""
        chunks = chunk_text(text)
        all_entities = []

        for chunk in chunks:
            try:
                results = ner_pipeline(chunk)
                for ent in results:
                    # Extract ALL entity types — no filter applied
                    all_entities.append({
                        "text": ent["word"].strip(),
                        "type": ent["entity_group"],
                        "score": round(ent["score"], 4)
                    })
            except Exception as e:
                logging.warning(f"NER chunk failed: {e}")
                continue

        # Deduplicate by (text, type)
        seen = set()
        unique_entities = []
        for ent in all_entities:
            key = (ent["text"].lower(), ent["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(ent)

        return unique_entities

    # ======================================================
    # 6. INFER WHICH FILES WERE PROCESSED
    # Uses the openai output folder as reference to avoid
    # running NER on the entire dataset.
    # ======================================================

    summary_files_ref = sorted(glob.glob(
        os.path.join(base_summaries_folder, "openai", "*.txt")
    ))

    if not summary_files_ref:
        logging.warning("No OpenAI summaries found — cannot infer processed files.")
        processed_names = []
    else:
        processed_names = [
            os.path.basename(f).replace("_openai.txt", ".txt")
            for f in summary_files_ref
        ]
        logging.info(f"Inferred {len(processed_names)} processed files from OpenAI summaries.")

    # ======================================================
    # 7. PROCESS ORIGINAL CLINICAL NOTES (only processed files)
    # ======================================================

    logging.info("Processing original clinical notes...")

    source_files = [
        os.path.join(dataset_folder, name)
        for name in processed_names
        if os.path.exists(os.path.join(dataset_folder, name))
    ]

    if not source_files:
        logging.warning(f"No matching source files found in {dataset_folder}")
    else:
        source_records = []

        for file_path in source_files:
            file_name = os.path.basename(file_path)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                entities = extract_entities(text)

                for ent in entities:
                    source_records.append({
                        "File": file_name,
                        "Source": "original_note",
                        "Entity_Text": ent["text"],
                        "Entity_Type": ent["type"],
                        "Confidence": ent["score"]
                    })

                logging.info(f"  {file_name}: {len(entities)} entities extracted")

            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")

        # Save original notes NER results
        source_csv = os.path.join(ner_folder, "ner_original_notes.csv")
        _save_csv(source_records, source_csv)
        logging.info(f"Original notes NER saved: {source_csv} ({len(source_records)} rows)")

    # ======================================================
    # 8. PROCESS GENERATED SUMMARIES (PER MODEL)
    # ======================================================

    for model in models_to_process:

        summaries_folder = os.path.join(base_summaries_folder, model)
        summary_files = sorted(glob.glob(os.path.join(summaries_folder, "*.txt")))

        if not summary_files:
            logging.warning(f"Skipping '{model}': No summaries found.")
            continue

        logging.info(f"Processing summaries: {model.upper()} ({len(summary_files)} files)")

        model_records = []

        for sum_file in summary_files:
            file_name = os.path.basename(sum_file)
            base_name = file_name.replace(f"_{model}.txt", ".txt")

            try:
                with open(sum_file, "r", encoding="utf-8") as f:
                    text = f.read()

                entities = extract_entities(text)

                for ent in entities:
                    model_records.append({
                        "File": base_name,
                        "Model": model.upper(),
                        "Source": "generated_summary",
                        "Entity_Text": ent["text"],
                        "Entity_Type": ent["type"],
                        "Confidence": ent["score"]
                    })

                logging.info(f"  {base_name} [{model.upper()}]: {len(entities)} entities extracted")

            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")

        # Save per-model NER results
        model_csv = os.path.join(ner_folder, f"ner_{model}_summaries.csv")
        _save_csv(model_records, model_csv)
        logging.info(f"{model.upper()} NER saved: {model_csv} ({len(model_records)} rows)")

    # ======================================================
    # 9. SAVE COMBINED CSV (all sources + models in one file)
    # ======================================================

    all_csv_files = glob.glob(os.path.join(ner_folder, "ner_*.csv"))
    combined_records = []

    for csv_path in sorted(all_csv_files):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                combined_records.append(row)

    if combined_records:
        combined_csv = os.path.join(ner_folder, "ner_combined.csv")
        _save_csv(combined_records, combined_csv)
        logging.info(f"Combined NER saved: {combined_csv} ({len(combined_records)} total rows)")

    logging.info("NER EXTRACTION COMPLETE.")

# ======================================================
# 10. HELPER: Save CSV
# ======================================================

def _save_csv(records: list, path: str):
    """Save a list of dicts to a CSV file."""
    if not records:
        logging.warning(f"No records to save for: {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

# ======================================================
# Run standalone for testing
# ======================================================
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if len(sys.argv) == 3:
        run_ner(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python ner_extraction.py <run_name> <timestamp>")