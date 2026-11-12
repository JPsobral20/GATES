"""
it3_graph.py
============
Implementation of Flow 3: Neuro-Symbolic Knowledge Graph Extraction.
Transforms raw clinical notes into chronological and auditable JSON structures.

Main Features:
- Grounding: Uses entities extracted in Flow 2 to prevent LLM hallucinations.
- Traceability: Requires an 'evidence' field to prove every extracted relationship.
- Stability: Forces OpenAI's 'JSON Mode' to ensure reliable parsing.
"""

import os
import glob
import time
import json
import logging
import argparse
import csv
from datetime import datetime
from dotenv import load_dotenv


from it1_baseline import setup_clients, call_with_retry
from it2_ner import extract_entities 


# PROMPT CONFIGURATION (Strict Ontology)
def build_graph_prompt(clinical_text: str, extracted_entities: list) -> str:
    """Builds the extraction prompt following strict ontology guidelines."""
    
    # Basic deduplication using Python set()
    unique_entities = list(set([ent["text"] for ent in extracted_entities]))
    entities_str = ", ".join(unique_entities)
    
    return f"""You are a clinical data extractor for a Scientific Knowledge Graph.
Extract relationships between the provided entities from the clinical text.

ENTITIES TO CONNECT:
[{entities_str}]

ALLOWED RELATIONS (Use ONLY these):
- "has_symptom" (Patient -> Symptom)
- "diagnosed_with" (Patient -> Condition)
- "prescribed_medication" (Patient/Condition -> Drug)
- "is_allergic_to" (Patient -> Drug/Substance)
- "has_adverse_reaction" (Drug -> Symptom)
- "underwent_procedure" (Patient -> Surgery/Test)
- "resulted_in" (Procedure/Test -> Outcome)
- "happened_before" (Event A -> Event B)

STRICT RULES:
1. ONLY connect the entities provided in the list above. Do not invent new entities.
2. Capture dates/times in the "date_time" field (use ISO format or relative time).
3. The "evidence" field MUST contain the exact sentence from the text proving this link.
4. Evaluate the context to classify the relationship in the "status" field. Use "Active" (current problems), "Resolved" (cured/past problems), or "Negated" (explicitly denied/no longer applicable).
5. Output ONLY valid JSON.

CLINICAL TEXT:
"{clinical_text}"

OUTPUT SCHEMA:
{{
  "graph": [
    {{
      "subject": "Entity A",
      "relation": "RELATION_NAME",
      "object": "Entity B",
      "status": "Active | Resolved | Negated",
      "date_time": "YYYY-MM-DD or '2 days ago'",
      "evidence": "Quote from text"
    }}
  ]
}}"""


# SINGLE-FILE PIPELINE (called by it3_summary.py)

def run_flow3_graph(file_path: str, client_openai, output_dir: str) -> list:
    """
    Processes a single file: runs NER and extracts a Knowledge Graph.
    :param file_path: Path to the clinical note .txt file.
    :param client_openai: Initialized OpenAI client.
    :param output_dir: Directory to save the individual graph JSON.
    :return: List of graph relations, or empty list on failure.
    """
    file_name = os.path.basename(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # NER Grounding (Fact Extraction from Flow 2)
    logging.info(f"Processing {file_name}: Extracting Entities via NER...")
    entities = extract_entities(text)
    
    if entities:
        ner_dir = os.path.join(output_dir, "ner_entities")
        os.makedirs(ner_dir, exist_ok=True)
        csv_path = os.path.join(ner_dir, file_name.replace(".txt", "_ner.csv"))
        
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=entities[0].keys())
            writer.writeheader()
            writer.writerows(entities)

    # LLM Inference (Relation Extraction in JSON Mode)
    prompt = build_graph_prompt(text, entities)

    def call_llm():
        return client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a clinical JSON extractor."},
                      {"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

    logging.info(f"Processing {file_name}: Generating Knowledge Graph...")
    response = call_with_retry(call_llm)

    if response:
        graph_json = json.loads(response.choices[0].message.content)

        # Save the individual graph
        os.makedirs(os.path.join(output_dir, "individual_graphs"), exist_ok=True)
        individual_path = os.path.join(output_dir, "individual_graphs", file_name.replace(".txt", ".json"))
        with open(individual_path, "w", encoding="utf-8") as j:
            json.dump(graph_json, j, indent=4, ensure_ascii=False)

        graph = graph_json.get("graph", [])
        logging.info(f"Success: {len(graph)} relations extracted.")
        return graph

    return []


# STANDALONE CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract JSON Knowledge Graphs from Clinical Notes")
    parser.add_argument("--files", type=int, default=3, help="Number of files to process")
    args = parser.parse_args()

    load_dotenv()
    try:
        client_openai, _, _ = setup_clients()
    except Exception as e:
        print(f"Setup error: {e}")
        exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #input_dir = "./dataset/dataset/"
    input_dir = "./dataset/edge_cases_test/"
    output_dir = f"./outputs/it3_graphs/{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(output_dir, "execution.log")), logging.StreamHandler()]
    )

    all_files = glob.glob(os.path.join(input_dir, "*.txt"))
    target_files = all_files[:args.files] if args.files > 0 else all_files

    master_timeline = {}
    logging.info(f"Starting Flow 3 (Graph Extraction) for {len(target_files)} files...")

    for file_path in target_files:
        file_name = os.path.basename(file_path)
        graph = run_flow3_graph(file_path, client_openai, output_dir)
        master_timeline[file_name] = graph

    master_path = os.path.join(output_dir, "master_timeline.json")
    with open(master_path, "w", encoding="utf-8") as m:
        json.dump(master_timeline, m, indent=4, ensure_ascii=False)

    logging.info(f"Pipeline Completed. Master Timeline saved at: {master_path}")