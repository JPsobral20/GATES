import os
import logging
import glob
import csv
import numpy as np
import tiktoken
import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from scipy.stats import ttest_rel
from transformers import AutoTokenizer

# ======================================================
# 1. CONFIGURATION
# ======================================================

dataset_folder = "./dataset/dataset/"
models_to_evaluate = ["openai", "nebius", "google"]

# Set True only if you have gold discharge summaries
USE_REFERENCES = False
reference_folder = "./references/"

tokenizer = tiktoken.get_encoding("cl100k_base")

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

BERTSCORE_BIOMED_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BERTSCORE_GENERAL_MODEL = "roberta-large"
BERT_MAX_TOKENS = 512  # BERT limit

# Pre-load tokenizers to avoid warnings (optional)
try:
    AutoTokenizer.from_pretrained(BERTSCORE_BIOMED_MODEL)
    AutoTokenizer.from_pretrained(BERTSCORE_GENERAL_MODEL)
except Exception:
    pass

# ======================================================
# MAIN ENTRY POINT
# ======================================================

def evaluate_run(run_name: str, timestamp: str, base_override: str = None):
    base_summaries_folder = base_override if base_override else f"./outputs/{run_name}/{timestamp}/"
    evaluation_folder = os.path.join(base_summaries_folder, "evaluation")
    os.makedirs(evaluation_folder, exist_ok=True)

    logging.info("=" * 60)
    logging.info("MULTI-MODEL EVALUATION STARTED")
    logging.info(f"Run: {run_name} | Timestamp: {timestamp}")
    logging.info(f"Biomed Model: {BERTSCORE_BIOMED_MODEL}")
    logging.info(f"General Model: {BERTSCORE_GENERAL_MODEL}")
    logging.info("=" * 60)

    # Load generation times
    generation_times = {}
    times_csv_path = os.path.join(base_summaries_folder, "generation_times.csv")
    if os.path.exists(times_csv_path):
        with open(times_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["Model"].upper(), row["File"] + ".txt")
                if "Overhead_sec" in row and "LLM_Time_sec" in row:
                    generation_times[key] = float(row["Overhead_sec"]) + float(row["LLM_Time_sec"])
                else:
                    generation_times[key] = float(row["Execution_Time_sec"])
        logging.info(f"Loaded generation times for {len(generation_times)} entries.")
    else:
        logging.warning("generation_times.csv not found.")

    def truncate_for_bert(text, max_tokens=BERT_MAX_TOKENS):
        tokens = tokenizer.encode(text)
        tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)

    # ======================================================
    # 2. INITIALIZE METRICS
    # ======================================================

    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth_func = SmoothingFunction().method1

    overall_averages_data = []
    bert_distributions = {} 

    # ======================================================
    # 3. PROCESS EACH MODEL
    # ======================================================

    for model in models_to_evaluate:
        summaries_folder = os.path.join(base_summaries_folder, model)
        summary_files = sorted(glob.glob(os.path.join(summaries_folder, "*.txt")))

        if not summary_files:
            continue

        logging.info(f"Evaluating model: {model.upper()} ({len(summary_files)} files)")

        all_references = []
        all_summaries = []
        file_records = []

        model_averages = {
            "exec_time": [], "comp": [], 
            "r1_p": [], "r1_r": [], "r1_f": [], 
            "r2_p": [], "r2_r": [], "r2_f": [], 
            "rL_p": [], "rL_r": [], "rL_f": [],
            "bleu": [], "meteor": []
        }

        skipped = 0

        for sum_file in summary_files:
            file_name = os.path.basename(sum_file)
            base_name = file_name.replace(f"_{model}.txt", ".txt")
            source_file = os.path.join(dataset_folder, base_name)

            if not os.path.exists(source_file):
                skipped += 1
                continue

            with open(source_file, "r", encoding="utf-8") as sf:
                source_text = sf.read()
            with open(sum_file, "r", encoding="utf-8") as sumf:
                summary_text = sumf.read()

            if USE_REFERENCES:
                ref_file = os.path.join(reference_folder, base_name)
                if os.path.exists(ref_file):
                    with open(ref_file, "r", encoding="utf-8") as rf:
                        reference_text = rf.read()
                else:
                    reference_text = source_text
            else:
                reference_text = source_text

            all_references.append(reference_text)
            all_summaries.append(summary_text)

            source_tokens = len(tokenizer.encode(source_text))
            summary_tokens = len(tokenizer.encode(summary_text))
            compression_ratio = summary_tokens / source_tokens if source_tokens > 0 else 0

            # ROUGE
            rouge_scores = rouge.score(reference_text, summary_text)
            r1_f = rouge_scores['rouge1'].fmeasure
            rL_f = rouge_scores['rougeL'].fmeasure

            # Store metrics
            model_averages["comp"].append(compression_ratio)
            model_averages["r1_p"].append(rouge_scores['rouge1'].precision)
            model_averages["r1_r"].append(rouge_scores['rouge1'].recall)
            model_averages["r1_f"].append(r1_f)
            model_averages["r2_p"].append(rouge_scores['rouge2'].precision)
            model_averages["r2_r"].append(rouge_scores['rouge2'].recall)
            model_averages["r2_f"].append(rouge_scores['rouge2'].fmeasure)
            model_averages["rL_p"].append(rouge_scores['rougeL'].precision)
            model_averages["rL_r"].append(rouge_scores['rougeL'].recall)
            model_averages["rL_f"].append(rL_f)

            # BLEU & METEOR
            ref_words = word_tokenize(reference_text)
            summary_words = word_tokenize(summary_text)
            bleu = sentence_bleu([ref_words], summary_words, smoothing_function=smooth_func)
            meteor = single_meteor_score(ref_words, summary_words)
            model_averages["bleu"].append(bleu)
            model_averages["meteor"].append(meteor)

            exec_time = generation_times.get((model.upper(), base_name), "N/A")
            if exec_time != "N/A":
                model_averages["exec_time"].append(float(exec_time))

            file_records.append({
                "Model": model.upper(),
                "File": base_name,
                "Execution_Time_sec": exec_time,
                "Compression_Ratio": round(compression_ratio, 4),
                "ROUGE_1_P": round(rouge_scores['rouge1'].precision, 4),
                "ROUGE_1_R": round(rouge_scores['rouge1'].recall, 4),
                "ROUGE_1_F1": round(r1_f, 4),
                "ROUGE_2_P": round(rouge_scores['rouge2'].precision, 4),
                "ROUGE_2_R": round(rouge_scores['rouge2'].recall, 4),
                "ROUGE_2_F1": round(rouge_scores['rouge2'].fmeasure, 4),
                "ROUGE_L_P": round(rouge_scores['rougeL'].precision, 4),
                "ROUGE_L_R": round(rouge_scores['rougeL'].recall, 4),
                "ROUGE_L_F1": round(rL_f, 4),
                "BLEU": round(bleu, 4),
                "METEOR": round(meteor, 4)
            })

        if not file_records:
            continue

        # ==================================================
        # 4. BERTScore (Biomed vs General)
        # ==================================================
        all_summaries_trunc = [truncate_for_bert(s) for s in all_summaries]
        all_references_trunc = [truncate_for_bert(r) for r in all_references]

        # --- BIOMED BERT ---
        logging.info("Calculating Biomed BERTScore...")
        try:
            P_bio, R_bio, F1_bio = bert_score(
                all_summaries_trunc, all_references_trunc, lang="en",
                model_type=BERTSCORE_BIOMED_MODEL, num_layers=12, verbose=False
            )
        except Exception as e:
            logging.error(f"BiomedBERT failed: {e}")
            P_bio, R_bio, F1_bio = np.zeros(len(file_records)), np.zeros(len(file_records)), np.zeros(len(file_records))

        # --- GENERAL BERT ---
        logging.info("Calculating General BERTScore...")
        try:
            P_gen, R_gen, F1_gen = bert_score(
                all_summaries_trunc, all_references_trunc, lang="en",
                model_type=BERTSCORE_GENERAL_MODEL, verbose=False
            )
        except Exception as e:
            logging.error(f"General BERT failed: {e}")
            P_gen, R_gen, F1_gen = np.zeros(len(file_records)), np.zeros(len(file_records)), np.zeros(len(file_records))

        bert_bio_p, bert_bio_r, bert_bio_f = P_bio.tolist(), R_bio.tolist(), F1_bio.tolist()
        bert_gen_p, bert_gen_r, bert_gen_f = P_gen.tolist(), R_gen.tolist(), F1_gen.tolist()

        bert_distributions[model] = {rec["File"]: score for rec, score in zip(file_records, bert_bio_f)}

        for i, record in enumerate(file_records):
            record["BERTScore_Biomed_P"] = round(bert_bio_p[i], 4)
            record["BERTScore_Biomed_R"] = round(bert_bio_r[i], 4)
            record["BERTScore_Biomed_F1"] = round(bert_bio_f[i], 4)
            record["BERTScore_Gen_P"] = round(bert_gen_p[i], 4)
            record["BERTScore_Gen_R"] = round(bert_gen_r[i], 4)
            record["BERTScore_Gen_F1"] = round(bert_gen_f[i], 4)

        # ==================================================
        # 5. Compute Averages & Save CSV
        # ==================================================
        avg_exec_time = np.mean(model_averages["exec_time"]) if model_averages["exec_time"] else "N/A"
        
        avg_bert_bio_f = np.mean(bert_bio_f)
        std_bert_bio_f = np.std(bert_bio_f)
        avg_bert_gen_f = np.mean(bert_gen_f)

        overall_averages_data.append({
            "Model": model.upper(),
            "Files_Evaluated": len(file_records),
            "Avg_Execution_Time_sec": round(avg_exec_time, 2) if avg_exec_time != "N/A" else "N/A",
            "Avg_Compression_Ratio": round(np.mean(model_averages['comp']), 4),
            "Avg_ROUGE_1_P": round(np.mean(model_averages['r1_p']), 4),
            "Avg_ROUGE_1_R": round(np.mean(model_averages['r1_r']), 4),
            "Avg_ROUGE_1_F1": round(np.mean(model_averages['r1_f']), 4),
            "Avg_ROUGE_2_P": round(np.mean(model_averages['r2_p']), 4),
            "Avg_ROUGE_2_R": round(np.mean(model_averages['r2_r']), 4),
            "Avg_ROUGE_2_F1": round(np.mean(model_averages['r2_f']), 4),
            "Avg_ROUGE_L_P": round(np.mean(model_averages['rL_p']), 4),
            "Avg_ROUGE_L_R": round(np.mean(model_averages['rL_r']), 4),
            "Avg_ROUGE_L_F1": round(np.mean(model_averages['rL_f']), 4),
            "Avg_BLEU": round(np.mean(model_averages['bleu']), 4),
            "Avg_METEOR": round(np.mean(model_averages['meteor']), 4),
            "Avg_BERTScore_Biomed_P": round(np.mean(bert_bio_p), 4),
            "Avg_BERTScore_Biomed_R": round(np.mean(bert_bio_r), 4),
            "Avg_BERTScore_Biomed_F1": round(avg_bert_bio_f, 4),
            "Std_BERTScore_Biomed_F1": round(std_bert_bio_f, 4),
            "Avg_BERTScore_Gen_P": round(np.mean(bert_gen_p), 4),
            "Avg_BERTScore_Gen_R": round(np.mean(bert_gen_r), 4),
            "Avg_BERTScore_Gen_F1": round(avg_bert_gen_f, 4),
            "Files_Skipped": skipped,
        })

        logging.info(f"Biomed F1: {avg_bert_bio_f:.4f} | General F1: {avg_bert_gen_f:.4f}")

        output_csv_path = os.path.join(evaluation_folder, f"{model}_evaluation_results.csv")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=file_records[0].keys())
            writer.writeheader()
            writer.writerows(file_records)

    # ======================================================
    # 6. STATISTICAL SIGNIFICANCE (Paired t-test for Biomed)
    # ======================================================
    logging.info("=" * 60)
    logging.info("STATISTICAL SIGNIFICANCE (Biomed BERTScore F1)")
    logging.info("=" * 60)
    model_names = list(bert_distributions.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            common_files = set(bert_distributions[m1].keys()).intersection(set(bert_distributions[m2].keys()))
            if len(common_files) < 2: continue
            scores_m1 = [bert_distributions[m1][f] for f in common_files]
            scores_m2 = [bert_distributions[m2][f] for f in common_files]
            t_stat, p_val = ttest_rel(scores_m1, scores_m2)
            logging.info(f"{m1.upper()} vs {m2.upper()} -> p = {p_val:.6f} (n={len(common_files)})")

    # ======================================================
    # 7. SAVE MASTER AVERAGES
    # ======================================================
    if overall_averages_data:
        averages_csv_path = os.path.join(evaluation_folder, "overall_averages.csv")
        with open(averages_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=overall_averages_data[0].keys())
            writer.writeheader()
            writer.writerows(overall_averages_data)
        logging.info("MULTI-MODEL EVALUATION COMPLETE.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        evaluate_run(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python evaluate_metrics.py <run_name> <timestamp>")