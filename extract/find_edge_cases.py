import os
import glob
import re
import csv
import time

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
DATASET_FOLDER = "../dataset/dataset/"
OUTPUT_CSV = "./candidatos_edge_cases.csv"

# Expressões Regulares para procurar padrões caóticos (Case Insensitive)
REGEX_NO_ALLERGY = re.compile(r'\b(nkda|no known drug allergies|no allergies)\b', re.IGNORECASE)
REGEX_ALLERGY = re.compile(r'\b(allergic to|severe reaction to)\b', re.IGNORECASE)

REGEX_NEG_FAMILY = re.compile(r'\b(patient denies|family history|mother died|father passed away)\b', re.IGNORECASE)

REGEX_NOISE = re.compile(r'\b(copyright|all rights reserved|cafeteria|menu|automated message)\b', re.IGNORECASE)

# Limiar de Note Bloat (Ficheiros com mais de 3500 palavras são massivos)
WORD_COUNT_THRESHOLD = 3500

def run_funnel_scanner():
    print("🔍 A iniciar o scanner heurístico de Edge Cases...")
    start_time = time.time()
    
    files = glob.glob(os.path.join(DATASET_FOLDER, "*.txt"))
    if not files:
        print(f"❌ Não foram encontrados ficheiros em {DATASET_FOLDER}")
        return

    print(f"📁 Encontrados {len(files)} ficheiros. A analisar...")
    
    candidates = []
    
    for count, file_path in enumerate(files, 1):
        if count % 5000 == 0:
            print(f"  ⏳ Processados {count}/{len(files)} ficheiros...")

        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            continue
            
        word_count = len(text.split())
        
        # 1. Teste de Redundância / Note Bloat Extremo
        if word_count > WORD_COUNT_THRESHOLD:
            candidates.append({
                "Ficheiro": file_name, 
                "Categoria": "Note Bloat (Redundância)", 
                "Palavras": word_count
            })
            
        # 2. Teste de Contradição (Alergias)
        if REGEX_NO_ALLERGY.search(text) and REGEX_ALLERGY.search(text):
            candidates.append({
                "Ficheiro": file_name, 
                "Categoria": "Contradição (Alergias)", 
                "Palavras": word_count
            })
            
        # 3. Teste de Negação / Histórico Familiar
        if REGEX_NEG_FAMILY.search(text):
            candidates.append({
                "Ficheiro": file_name, 
                "Categoria": "Negação / Hist Fam.", 
                "Palavras": word_count
            })
            
        # 4. Teste de Ruído Não-Clínico
        if REGEX_NOISE.search(text):
            candidates.append({
                "Ficheiro": file_name, 
                "Categoria": "Ruído (Boilerplate)", 
                "Palavras": word_count
            })

    # Guardar os resultados num CSV
    if candidates:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Ficheiro", "Categoria", "Palavras"])
            writer.writeheader()
            writer.writerows(candidates)
            
        print(f"\n✅ Concluído em {round(time.time() - start_time, 2)} segundos!")
        print(f"🎯 Foram encontrados {len(candidates)} potenciais Edge Cases.")
        print(f"📂 Resultados guardados em: {OUTPUT_CSV}")
    else:
        print("\n🧐 O scanner terminou, mas não encontrou candidatos fortes. O dataset parece estar muito limpo.")

if __name__ == "__main__":
    run_funnel_scanner()