"""
baseline.py
===========
Executes the baseline (Control Flow) for Clinical Summarization.
Uses basic prompt engineering without structural anchoring (No NER/Graphs).

Can be run standalone via CLI or imported as a module in it2_ner.py.
"""

import os
import glob
import time
import csv
import logging
import argparse
from datetime import datetime
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Only importing the evaluation metrics, NO NER.
from evaluate_metrics import evaluate_run


# SETUP and AUTHENTICATION
def setup_clients():
    """Loads environment variables and initializes API clients."""
    load_dotenv()
    
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")

    missing_keys = []
    if not OPENAI_API_KEY: missing_keys.append("OPENAI_API_KEY")
    if not GOOGLE_API_KEY: missing_keys.append("GOOGLE_API_KEY")
    if not NEBIUS_API_KEY: missing_keys.append("NEBIUS_API_KEY")

    if missing_keys:
        raise ValueError(f"Missing API keys in .env file: {', '.join(missing_keys)}")
        
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    client_nebius = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=NEBIUS_API_KEY,
    )
    genai.configure(api_key=GOOGLE_API_KEY)
    model_google = genai.GenerativeModel('gemini-2.5-flash')
    
    return client_openai, client_nebius, model_google

# PROMPT BUILDER
def build_prompt(clinical_note: str) -> str:
    """Builds the standard baseline prompt for the clinical note."""
    return f"""You are an expert medical scribe. Read the following hospital course note and write a concise Discharge Summary.

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

Write the Discharge Summary now. Be concise but complete. Do not add commentary outside the sections above."""


# RETRY HELPER
def call_with_retry(fn, retries=3, delay=5):
    """Executes an API call with automatic retries on failure."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            logging.warning(f"Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)  
    return None


# MAIN PIPELINE
def run_baseline(num_files_to_process: int = 0, run_name: str = "it1_control", timestamp: str = None, parent_folder: str = None):
    """
    Executes the baseline pipeline.
    :param num_files_to_process: Number of files to process. 0 processes all.
    :param run_name: Name of the run for folder organization.
    :param timestamp: Optional timestamp to sync with other scripts.
    :param parent_folder: Optional parent folder to save results into a shared directory.
    :return: The final output directory path.
    """
    try:
        client_openai, client_nebius, model_google = setup_clients()
    except Exception as e:
        print(f"Setup Error: {e}")
        return None

    # Use existing timestamp or create a new one if running standalone
    actual_timestamp = timestamp if timestamp else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #input_folder = "./dataset/dataset/"
    input_folder = "./dataset/edge_cases_test/"
    
    # Logic to define the correct output path
    if parent_folder:
        # Synced execution (e.g., inside it2_ner.py)
        output_base = os.path.join(parent_folder, run_name)
    else:
        # Standalone execution
        output_base = f"./outputs/{run_name}/{actual_timestamp}/"

    os.makedirs(os.path.join(output_base, "openai"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "google"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "nebius"), exist_ok=True)

    # --- LOGGING SETUP ---
    log_file_path = os.path.join(output_base, f"execution_{actual_timestamp}.log")
    
    # Remove old handlers to prevent duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    files = glob.glob(os.path.join(input_folder, "*.txt"))

    if not files:
        logging.error(f"Could not find any .txt files in {input_folder}")
        return None
        
    target_files = files[:num_files_to_process] if num_files_to_process > 0 else files

    logging.info(f"Found {len(files)} files. Processing {len(target_files)} files.")
    logging.info(f"Starting baseline generation for run: {run_name} at {actual_timestamp}...\n")

    time_records = []
    system_msg = "You are a helpful medical assistant. Follow all formatting instructions exactly."

    # Timer before processing each file to capture overhead and LLM time separately
    timer_before_file = time.time() - global_start

    for file_path in target_files: 
        file_start_time = time.time()
        file_name = os.path.basename(file_path)
        logging.info(f"Processing: {file_name}")

        with open(file_path, "r", encoding="utf-8") as file:
            clinical_note = file.read()

        prompt = build_prompt(clinical_note)

        pre_llm_overhead = time.time() - file_start_time + timer_before_file

        # OPENAI (GPT-4o-Mini)
        def call_openai():
            return client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                temperature=0.0
            )

        start_time = time.time()
        response_openai = call_with_retry(call_openai)
        exec_time_openai = time.time() - start_time

        if response_openai:
            summary = response_openai.choices[0].message.content
            with open(os.path.join(output_base, "openai", file_name.replace(".txt", "_openai.txt")), "w", encoding="utf-8") as f:
                f.write(summary)
            time_records.append({"File": file_name.replace(".txt", ""), "Model": "OPENAI", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(exec_time_openai, 2)})
        else:
            logging.error(f"OpenAI failed after all retries for {file_name}")

        # NEBIUS (Llama-3.3-70B-Instruct)
        def call_nebius():
            return client_nebius.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
                temperature=0.0
            )

        start_time = time.time()
        response_nebius = call_with_retry(call_nebius)
        exec_time_nebius = time.time() - start_time

        if response_nebius:
            summary = response_nebius.choices[0].message.content
            with open(os.path.join(output_base, "nebius", file_name.replace(".txt", "_nebius.txt")), "w", encoding="utf-8") as f:
                f.write(summary)
            time_records.append({"File": file_name.replace(".txt", ""), "Model": "NEBIUS", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(exec_time_nebius, 2)})
        else:
            logging.error(f"Nebius failed after all retries for {file_name}")

        # GOOGLE (Gemini 2.5 Flash)
        def call_google():
            return model_google.generate_content(
                f"System: {system_msg}\n\nUser: {prompt}",
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )

        start_time = time.time()
        response_google = call_with_retry(call_google)
        exec_time_google = time.time() - start_time

        if response_google and hasattr(response_google, "text") and response_google.text:
            summary = response_google.text
            with open(os.path.join(output_base, "google", file_name.replace(".txt", "_google.txt")), "w", encoding="utf-8") as f:
                f.write(summary)
            time_records.append({"File": file_name.replace(".txt", ""), "Model": "GOOGLE", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(exec_time_google, 2)})
        else:
            logging.error(f"Google failed after all retries for {file_name}")

        logging.info(f"Done: {file_name}")

    # --- SAVE EXECUTION TIMES ---
    time_csv_path = os.path.join(output_base, "generation_times.csv")
    if time_records:
        with open(time_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["File", "Model", "Overhead_sec", "LLM_Time_sec"])
            writer.writeheader()
            writer.writerows(time_records)

    logging.info(f"Baseline generation complete. Results saved to: {output_base}")

    # --- AUTOMATIC EVALUATION (NO NER) ---
    logging.info("Starting automatic evaluation pipeline...")
    # Passed base_override so it finds the correct synchronized folder
    evaluate_run(run_name, actual_timestamp, base_override=output_base)
    
    return output_base


# COMMAND LINE INTERFACE

if __name__ == "__main__":
    global_start = time.time()
    parser = argparse.ArgumentParser(description="Baseline Clinical Summarization using LLMs")
    
    parser.add_argument(
        "--files", 
        type=int, 
        default=3, 
        help="Number of files to process. Set to 0 to process ALL files in the dataset. Default is 3."
    )
    
    parser.add_argument(
        "--name", 
        type=str, 
        default="it1_baseline", 
        help="Run name for the output folder. Default is 'it1_baseline'."
    )
    
    args = parser.parse_args()
    
    # Execute the main pipeline with CLI parameters
    run_baseline(num_files_to_process=args.files, run_name=args.name)
    logging.info(f"Total pipeline time: {round(time.time() - global_start, 2)}s")