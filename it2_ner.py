"""
it2_ner.py
==========
Implementation of Iteration 2: NER Grounding.
Extracts biomedical entities using a local BERT model and forces the LLMs
to include these entities in the generated summaries.
"""

import os
import glob
import time
import csv
import logging
import argparse
from datetime import datetime
import nltk

from transformers import pipeline, AutoTokenizer

from it1_baseline import setup_clients, call_with_retry
from evaluate_metrics import evaluate_run

# Ensure NLTK resources are available for chunking
nltk.download('punkt', quiet=True)


# NER SETUP AND HELPERS (SENTENCE-BASED)
NER_MODEL = "d4data/biomedical-ner-all"
NER_MAX_TOKENS = 500

try:
    logging.info("Loading NER model...")
    ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
    ner_pipeline_instance = pipeline(
        "ner", model=NER_MODEL, tokenizer=ner_tokenizer,
        aggregation_strategy="first", device=-1
    )
except Exception as e:
    logging.error(f"Error loading NER model: {e}")

def chunk_text_by_sentence(text: str, max_tokens: int = NER_MAX_TOKENS) -> list:
    """Divides text into chunks respecting sentence boundaries."""
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0
    
    for sentence in sentences:
        sentence_tokens = len(ner_tokenizer.encode(sentence, add_special_tokens=False))
        if current_len + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_len += sentence_tokens
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extract_entities(text: str) -> list:
    """Extracts and deduplicates biomedical entities from text."""
    chunks = chunk_text_by_sentence(text)
    all_entities = []
    
    for chunk in chunks:
        results = ner_pipeline_instance(chunk)
        for ent in results:
            all_entities.append({
                "text": ent["word"].strip(), 
                "type": ent["entity_group"], 
                "score": ent["score"]
            })
    
    # Deduplication
    seen, unique = set(), []
    for ent in all_entities:
        key = (ent["text"].lower(), ent["type"])
        if key not in seen:
            seen.add(key)
            unique.append(ent)
            
    return unique


# PROMPT BUILDER
SYSTEM_MSG_NER = "You are a helpful medical assistant. Follow all formatting instructions exactly. Preserve all listed entities."
 
def build_prompt_with_entities(clinical_note: str, entities_list: list) -> str:
    """Builds the generation prompt anchoring the LLM to extracted entities."""
    grouped = {}
    for ent in entities_list:
        grouped.setdefault(ent["type"], []).append(ent["text"])
 
    entity_block = ""
    for etype, terms in grouped.items():
        entity_block += f"- {etype}: {', '.join(list(dict.fromkeys(terms)))}\n"
 
    return f"""You are an expert medical scribe. The following biomedical entities have been automatically identified in the clinical note below. You MUST preserve all of these entities in your summary — do not omit, alter, or fabricate any medical terms.
 
Identified Entities (MUST be preserved):
{entity_block}

Now read the following clinical note and write a concise Discharge Summary.
 
Your summary MUST follow this exact structure:
 
**Principal Diagnosis:**
[Primary reason for admission]
 
**Active Problems:**
[Numbered list of active medical problems addressed during this admission]
 
**Medications on Discharge:**
[Numbered list of medications with dose and frequency]
 
**Follow-up Plan:**
[Specific follow-up instructions, appointments, and pending results]
 
**Additional Notes:**
[Any other clinically relevant information]
 
Clinical Note:
{clinical_note}
 
Write the Discharge Summary now. Be concise but complete. Do not add commentary outside the sections above. Ensure every entity listed above appears in the appropriate section."""


# PIPELINE EXECUTION
def run_iteration2_ner(num_files_to_process: int = 3, run_name: str = "it2_ner"):
    try:
        client_openai, client_nebius, model_google = setup_clients()
    except Exception as e:
        print(f"Setup error: {e}")
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #input_folder = "./dataset/dataset/"
    input_folder = "./dataset/edge_cases_test/"
    output_base = f"./outputs/{run_name}/{timestamp}/"
    
    model_paths = {
        "openai": os.path.join(output_base, "openai"),
        "nebius": os.path.join(output_base, "nebius"),
        "google": os.path.join(output_base, "google")
    }
    for path in model_paths.values():
        os.makedirs(path, exist_ok=True)
        
    log_file_path = os.path.join(output_base, "execution.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file_path, encoding="utf-8"), logging.StreamHandler()])

    files = glob.glob(os.path.join(input_folder, "*.txt"))
    target_files = files[:num_files_to_process] if num_files_to_process > 0 else files

    if not target_files:
        logging.error("No files found to process.")
        return

    logging.info(f">>> STARTING ITERATION 2 (NER GROUNDING) for {len(target_files)} files <<<")
    time_records = []

    # Timer before processing each file to capture overhead and LLM time separately
    timer_before_file = time.time() - global_start

    for file_path in target_files:
        file_start_time = time.time()
        file_name = os.path.basename(file_path)
        logging.info(f"Processing: {file_name}")
        
        with open(file_path, "r", encoding="utf-8") as f: 
            clinical_note = f.read()

        logging.info(f"Extracting entities for {file_name}...")
        entities = extract_entities(clinical_note)
        prompt = build_prompt_with_entities(clinical_note, entities)

        pre_llm_overhead = time.time() - file_start_time + timer_before_file

        # OPENAI (GPT-4o-Mini)
        t = time.time()
        res_openai = call_with_retry(lambda: client_openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "system", "content": SYSTEM_MSG_NER}, {"role": "user", "content": prompt}], 
            temperature=0.0
        ))
        if res_openai:
            with open(os.path.join(model_paths["openai"], file_name.replace(".txt", "_openai.txt")), "w", encoding="utf-8") as f: 
                f.write(res_openai.choices[0].message.content)
            time_records.append({"File": file_name.replace(".txt", ""), "Model": "OPENAI", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(time.time()-t, 2)})

        # NEBIUS (Llama-3.3-70B-Instruct)
        t = time.time()
        res_nebius = call_with_retry(lambda: client_nebius.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct", 
            messages=[{"role": "system", "content": SYSTEM_MSG_NER}, {"role": "user", "content": prompt}], 
            temperature=0.0
        ))
        if res_nebius:
            with open(os.path.join(model_paths["nebius"], file_name.replace(".txt", "_nebius.txt")), "w", encoding="utf-8") as f: 
                f.write(res_nebius.choices[0].message.content)
            time_records.append({"File": file_name.replace(".txt", ""), "Model": "NEBIUS", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(time.time()-t, 2)})

        # GOOGLE (Gemini 2.5 Flash)
        t = time.time()
        res_google = call_with_retry(lambda: model_google.generate_content(
            f"System: {SYSTEM_MSG_NER}\n\nUser: {prompt}", 
            generation_config={"temperature": 0.0}
        ))
        if res_google and hasattr(res_google, "text"):
            with open(os.path.join(model_paths["google"], file_name.replace(".txt", "_google.txt")), "w", encoding="utf-8") as f: 
                f.write(res_google.text)
            time_records.append({"File": file_name.replace(".txt", ""), "Model": "GOOGLE", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(time.time()-t, 2)})


    with open(os.path.join(output_base, "generation_times.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["File", "Model", "Overhead_sec", "LLM_Time_sec"])
        writer.writeheader()
        writer.writerows(time_records)


    logging.info(">>> STARTING AUTOMATED EVALUATION <<<")
    evaluate_run(run_name, timestamp, base_override=output_base)
    
    logging.info(f">>> ITERATION 2 PIPELINE COMPLETE. ALL DATA SAVED IN: {output_base}")

if __name__ == "__main__":
    global_start = time.time()
    parser = argparse.ArgumentParser(description="Iteration 2: NER Grounded Summarization")
    parser.add_argument("--files", type=int, default=3, help="Number of files to process")
    args = parser.parse_args()
    
    run_iteration2_ner(args.files)
    logging.info(f"Total pipeline time: {round(time.time() - global_start, 2)}s")