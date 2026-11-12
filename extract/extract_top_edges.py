import pandas as pd
import os
import shutil

# ==========================================
# CONFIGURAÇÕES
# ==========================================
CSV_PATH = "./candidatos_edge_cases.csv"
ORIGIN_FOLDER = "../dataset/dataset/"
DEST_FOLDER = "../dataset/edge_cases_test/"

def extrair_elite_do_caos():
    # Criar a pasta de destino se não existir
    os.makedirs(DEST_FOLDER, exist_ok=True)

    print("📊 A ler o ficheiro CSV de candidatos...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Erro: O ficheiro {CSV_PATH} não foi encontrado.")
        return

    ficheiros_selecionados = []

    # 1. Os 2 piores casos de Note Bloat (os maiores ficheiros absolutos)
    bloat_df = df[df['Categoria'] == 'Note Bloat (Redundância)'].sort_values(by='Palavras', ascending=False)
    ficheiros_selecionados.extend(bloat_df.head(2)['Ficheiro'].tolist())

    # 2. Selecionar 2 casos fortes das restantes categorias
    outras_categorias = ['Contradição (Alergias)', 'Negação / Hist Fam.', 'Ruído (Boilerplate)']

    for cat in outras_categorias:
        cat_df = df[df['Categoria'] == cat]
        if not cat_df.empty:
            # Pegamos os maiores dessa categoria para garantir que o erro está enterrado no meio de muito texto
            ficheiros_selecionados.extend(cat_df.sort_values(by='Palavras', ascending=False).head(2)['Ficheiro'].tolist())

    # Remover possíveis duplicados (caso um ficheiro ganhe em duas categorias)
    ficheiros_selecionados = list(set(ficheiros_selecionados))

    print(f"🎯 Foram selecionados {len(ficheiros_selecionados)} ficheiros extremos para teste.")
    print("A copiar ficheiros...")

    # Copiar os ficheiros selecionados para a nova pasta
    copiados = 0
    for ficheiro in ficheiros_selecionados:
        origem = os.path.join(ORIGIN_FOLDER, ficheiro)
        destino = os.path.join(DEST_FOLDER, ficheiro)
        
        if os.path.exists(origem):
            shutil.copy2(origem, destino)
            print(f"  ✅ Copiado: {ficheiro}")
            copiados += 1
        else:
            print(f"  ⚠️ Não encontrado na origem: {origem}")

    print(f"\n🚀 Sucesso! {copiados} ficheiros isolados na pasta: {DEST_FOLDER}")

if __name__ == "__main__":
    extrair_elite_do_caos()