"""
it3_summary.py
==============
Multi-Model Implementation of Iteration 3: End-to-End Pipeline.
Orchestrates Knowledge Graph Extraction (via it3_graph.py) 
and generates Narrative Summaries using OpenAI, Google, and Nebius.
"""

import os
import glob
import json
import logging
import argparse
import time
import csv
from datetime import datetime

# Import API clients, retry logic, and the Graph engine
from it1_baseline import setup_clients, call_with_retry
from it3_graph import run_flow3_graph
from evaluate_metrics import evaluate_run


# PROMPT CONFIGURATION (Graph-to-Text)
def build_summary_prompt(graph_data: list) -> str:
    """Builds the prompt forcing the LLM to write ONLY based on the JSON Graph."""
    graph_string = json.dumps(graph_data, indent=2, ensure_ascii=False)
    
    return f"""You are an expert medical scribe. You are provided with a strict, validated Knowledge Graph containing a patient's medical history.

Your task is to write a clear, concise, and professional narrative Discharge Summary based ONLY on the facts present in this JSON Graph.

STRICT RULES:
1. NO HALLUCINATIONS: Do not add any medical conditions, medications, or events that are not explicitly present in the JSON.
2. NO BLOAT: If a concept is only mentioned once in the Graph, mention it only once in the text. Be direct.
3. If the graph contains chronological data, ensure the summary presents the events in chronological order.
4. STATUS AWARENESS: Pay close attention to the "status" field. If a relation is marked as "Resolved" or "Negated", you MUST explicitly state in the summary that the patient no longer has this condition or allergy.

JSON KNOWLEDGE GRAPH:
{graph_string}

Please structure your summary exactly like this:

**1. Patient Overview & Timeline**
[Chronological narrative of what happened based on 'happened_before' or dates]

**2. Diagnoses & Symptoms**
[Based on 'diagnosed_with' and 'has_symptom']

**3. Procedures & Outcomes**
[Based on 'underwent_procedure' and 'resulted_in']

**4. Medications**
[Based on 'prescribed_medication']
"""


# PROCESSING PIPELINE
def run_iteration3_pipeline(num_files: int, run_name: str = "it3_summaries"):
    # Setup all API Clients (OpenAI, Nebius, Google)
    try:
        client_openai, client_nebius, model_google = setup_clients()
    except Exception as e:
        print(f"Setup error: {e}")
        return

    # Setup output directories for Iteration 3
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"./outputs/{run_name}/{timestamp}/"
    graphs_dir = f"./outputs/it3_graphs/{timestamp}/"

    model_paths = {
        "openai": os.path.join(output_dir, "openai"),
        "nebius": os.path.join(output_dir, "nebius"),
        "google": os.path.join(output_dir, "google")
    }
    for path in model_paths.values():
        os.makedirs(path, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # Logging setup
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(output_dir, "execution.log")), logging.StreamHandler()]
    )

    #input_dir = "./dataset/dataset/"
    input_dir = "./dataset/edge_cases_test/"
    all_files = glob.glob(os.path.join(input_dir, "*.txt"))
    target_files = all_files[:num_files] if num_files > 0 else all_files

    if not target_files:
        logging.error("No files found to process.")
        return

    logging.info(f">>> STARTING ITERATION 3 for {len(target_files)} files <<<")

    system_msg = "You are a precise clinical summarizer based strictly on Knowledge Graphs."
    time_records = []
    master_timeline = {}
    
    # Timer before processing each file to capture overhead and LLM time separately
    timer_before_file = time.time() - global_start

    for file_path in target_files:
        # Timer starts here — captures NER + graph extraction + prompt build
        file_start_time = time.time()
        
        file_name = os.path.basename(file_path)
        base_filename = file_name.replace(".txt", "")
        logging.info(f"Processing: {file_name}")        

        # NER + Graph extraction for this file
        graph_data = run_flow3_graph(file_path, client_openai, graphs_dir)
        master_timeline[file_name] = graph_data

        if not graph_data:
            logging.warning(f"Skipping {file_name}: Empty graph data.")
            continue

        # Build summary prompt
        prompt = build_summary_prompt(graph_data)
        pre_llm_overhead = time.time() - file_start_time + timer_before_file

        # OPENAI (GPT-4o-Mini)
        logging.info(f"Calling OpenAI for {file_name}...")
        t_start = time.time()
        res_openai = call_with_retry(lambda: client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.0
        ))
        t_exec = time.time() - t_start
        if res_openai:
            with open(os.path.join(model_paths["openai"], f"{base_filename}_openai.txt"), "w", encoding="utf-8") as f:
                f.write(res_openai.choices[0].message.content)
            time_records.append({"File": base_filename, "Model": "OPENAI", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(t_exec, 2)})

        # NEBIUS (Llama-3.3-70B)
        logging.info(f"Calling Nebius for {file_name}...")
        t_start = time.time()
        res_nebius = call_with_retry(lambda: client_nebius.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.0
        ))
        t_exec = time.time() - t_start
        if res_nebius:
            with open(os.path.join(model_paths["nebius"], f"{base_filename}_nebius.txt"), "w", encoding="utf-8") as f:
                f.write(res_nebius.choices[0].message.content)
            time_records.append({"File": base_filename, "Model": "NEBIUS", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(t_exec, 2)})

        # GOOGLE (Gemini 2.5 Flash)
        logging.info(f"Calling Google for {file_name}...")
        t_start = time.time()
        res_google = call_with_retry(lambda: model_google.generate_content(
            f"System: {system_msg}\n\nUser: {prompt}",
            generation_config={"temperature": 0.0}
        ))
        t_exec = time.time() - t_start
        if res_google and hasattr(res_google, "text"):
            with open(os.path.join(model_paths["google"], f"{base_filename}_google.txt"), "w", encoding="utf-8") as f:
                f.write(res_google.text)
            time_records.append({"File": base_filename, "Model": "GOOGLE", "Overhead_sec": round(pre_llm_overhead, 2), "LLM_Time_sec": round(t_exec, 2)})

        logging.info(f"Done: {file_name}")

    # Save master timeline
    master_path = os.path.join(graphs_dir, "master_timeline.json")
    with open(master_path, "w", encoding="utf-8") as m:
        json.dump(master_timeline, m, indent=4, ensure_ascii=False)

    # Save generation times
    if time_records:
        with open(os.path.join(output_dir, "generation_times.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["File", "Model", "Overhead_sec", "LLM_Time_sec"])
            writer.writeheader()
            writer.writerows(time_records)

    logging.info(">>> STARTING AUTOMATED EVALUATION <<<")
    evaluate_run(run_name, timestamp, base_override=output_dir)
    logging.info(f">>> ITERATION 3 PIPELINE COMPLETE. Data saved in: {output_dir}")

if __name__ == "__main__":
    global_start = time.time()
    parser = argparse.ArgumentParser(description="Iteration 3 End-to-End Pipeline: Graph + Summary")
    parser.add_argument("--files", type=int, default=3, help="Number of files to process")
    args = parser.parse_args()

    run_iteration3_pipeline(args.files)
    logging.info(f"Total pipeline time: {round(time.time() - global_start, 2)}s")