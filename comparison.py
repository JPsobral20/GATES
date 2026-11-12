"""
comparison.py (DEBUG MODE)
==========================
Compares textual quality metrics and prints exactly what it finds 
to help diagnose missing models (OpenAI/Nebius).
"""

import os
import glob
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s") # Simplified logs for reading

def load_all_evaluations(run_dir: str, run_name: str) -> pd.DataFrame:
    eval_folder = run_dir if os.path.basename(run_dir) == "evaluation" else os.path.join(run_dir, "evaluation")
        
    if not os.path.exists(eval_folder):
        logging.error(f"❌ ERROR: Evaluation folder not found in {run_dir}")
        return pd.DataFrame()

    # Get all CSVs except time tracking and significance
    csv_files = [f for f in glob.glob(os.path.join(eval_folder, "*.csv")) 
                 if "generation_times" not in os.path.basename(f).lower() 
                 and "significance" not in os.path.basename(f).lower()]

    logging.info(f"📁 {run_name.upper()} - Found CSVs: {[os.path.basename(f) for f in csv_files]}")

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Ensure 'Model' column exists
            if 'Model' not in df.columns:
                # Try to guess from filename
                base_name = os.path.basename(file).lower()
                for m in ['openai', 'google', 'nebius']:
                    if m in base_name:
                        df['Model'] = m.upper()
                        break
                if 'Model' not in df.columns:
                    df['Model'] = 'UNKNOWN'
                    
            df['Model'] = df['Model'].astype(str).str.upper()
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop empty rows just in case
        combined_df = combined_df.dropna(subset=['Model', 'File'])
        logging.info(f"🤖 {run_name.upper()} - Models loaded: {combined_df['Model'].unique().tolist()}")
        return combined_df
        
    return pd.DataFrame()

def run_comparison(path1: str, path2: str, name1: str, name2: str, output_base: str):
    print("\n" + "="*60)
    print("🔍 DIAGNOSTIC MODE: LOADING DATA")
    print("="*60)
    
    df1 = load_all_evaluations(path1, name1)
    df2 = load_all_evaluations(path2, name2)

    if df1.empty or df2.empty:
        logging.error("Cannot proceed. Missing data in one of the runs.")
        return

    # Find common models
    models1 = set(df1['Model'].unique())
    models2 = set(df2['Model'].unique())
    models_present = models1.intersection(models2)

    print("\n" + "="*60)
    print("⚖️  COMPARISON RESULTS")
    print("="*60)
    logging.info(f"Models in {name1}: {models1}")
    logging.info(f"Models in {name2}: {models2}")
    logging.info(f"Models in COMMON (will be compared): {models_present}")
    
    if not models_present:
        logging.error("No models in common between the two runs! Cannot compare.")
        return

    non_metrics = ['Model', 'File']
    metrics = [col for col in df1.columns if col not in non_metrics and pd.api.types.is_numeric_dtype(df1[col])]

    per_file_records, model_summary_records, ttest_records = [], [], []

    for model in sorted(models_present):
        print(f"\n▶ Evaluating Model: {model}")
        m_df1 = df1[df1['Model'] == model]
        m_df2 = df2[df2['Model'] == model]

        merged = pd.merge(m_df1, m_df2, on='File', suffixes=(f'_{name1}', f'_{name2}'))
        
        if merged.empty:
            logging.warning(f"  No common files to compare for {model}.")
            continue

        logging.info(f"  Files paired: {len(merged)}")

        for _, row in merged.iterrows():
            record = {"Model": model, "File": row['File']}
            for m in metrics:
                v1, v2 = row.get(f"{m}_{name1}", np.nan), row.get(f"{m}_{name2}", np.nan)
                record[f"{name1}_{m}"] = v1
                record[f"{name2}_{m}"] = v2
                record[f"Delta_{m}"] = v2 - v1 if pd.notna(v1) and pd.notna(v2) else "N/A"
            per_file_records.append(record)

        summary = {"Model": model, "Files_Compared": len(merged)}
        
        for m in metrics:
            aligned_df = merged[[f"{m}_{name1}", f"{m}_{name2}"]].dropna()
            if aligned_df.empty: continue
            
            v1_aligned, v2_aligned = aligned_df[f"{m}_{name1}"], aligned_df[f"{m}_{name2}"]
            mean_v1, mean_v2 = v1_aligned.mean(), v2_aligned.mean()
            
            summary[f"{name1}_Avg_{m}"] = round(mean_v1, 4)
            summary[f"{name2}_Avg_{m}"] = round(mean_v2, 4)
            delta = mean_v2 - mean_v1
            summary[f"Delta_Avg_{m}"] = round(delta, 4)
            summary[f"Improvement_%_{m}"] = round((delta / mean_v1) * 100, 2) if mean_v1 != 0 else "N/A"

            if len(v1_aligned) >= 2:
                t_val, p_val = ttest_rel(v1_aligned, v2_aligned)
                sig = "YES" if p_val < 0.05 else "NO"
                ttest_records.append({
                    "Model": model, "Metric": m, "N_Pairs": len(v1_aligned),
                    f"{name1}_Mean": round(mean_v1, 4), f"{name2}_Mean": round(mean_v2, 4),
                    "Delta_Mean": round(delta, 4), "Improvement_%": summary[f"Improvement_%_{m}"],
                    "t_statistic": round(t_val, 4), "p_value": round(p_val, 6), "Significant_p005": sig
                })
                
                # Highlight key metrics in terminal
                if "BERTScore_F1" in m or "ROUGE_1_F1" in m:
                    print(f"    {m}: Improvement = {summary[f'Improvement_%_{m}']:+.2f}% | p-value = {p_val:.4f} ({sig})")

        model_summary_records.append(summary)

    os.makedirs(output_base, exist_ok=True)
    pd.DataFrame(per_file_records).to_csv(os.path.join(output_base, "comparison_per_file.csv"), index=False)
    pd.DataFrame(model_summary_records).to_csv(os.path.join(output_base, "comparison_model_summary.csv"), index=False)
    pd.DataFrame(ttest_records).to_csv(os.path.join(output_base, "comparison_ttest.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"✅ RESULTS SAVED TO: {output_base}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic Metrics Comparison")
    parser.add_argument("--path1", type=str, required=True)
    parser.add_argument("--path2", type=str, required=True)
    parser.add_argument("--name1", type=str, default="IT1")
    parser.add_argument("--name2", type=str, default="IT2")
    parser.add_argument("--out", type=str, default="./outputs/comparisons")
    
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = os.path.join(args.out, f"{args.name1}_vs_{args.name2}_{timestamp}")
    
    run_comparison(args.path1, args.path2, args.name1, args.name2, run_output_dir)