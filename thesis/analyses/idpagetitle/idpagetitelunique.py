import pandas as pd

# Charger le fichier CSV (ajustez le chemin et le séparateur si nécessaire)
file_path = '~/Desktop/projet_la/bdd/bdd.csv'
df = pd.read_csv(file_path, sep=';', encoding='UTF-8-SIG', low_memory=False)

# Liste pour stocker temporairement les DataFrames issus de chaque paire de colonnes
dfs = []

# Parcourir les 218 paires de colonnes
for n in range(218):
    col_title = f"pageTitle (actionDetails {n})"
    col_id = f"pageIdAction (actionDetails {n})"
    
    if col_title in df.columns and col_id in df.columns:
        temp_df = df[[col_title, col_id]].dropna(subset=[col_title])
        temp_df = temp_df.rename(columns={col_title: "pageTitle", col_id: "pageIdAction"})
        dfs.append(temp_df)

# Concaténer tous les DataFrames pour obtenir un seul tableau
all_df = pd.concat(dfs, ignore_index=True)

# Regrouper par pageTitle et agréger les identifiants associés en une chaîne séparée par des virgules
def aggregate_ids(ids):
    # Convertir en chaîne et conserver les valeurs uniques
    unique_ids = sorted(set(str(x) for x in ids if pd.notna(x)))
    return ", ".join(unique_ids)

grouped_df = all_df.groupby("pageTitle", as_index=False)["pageIdAction"].agg(aggregate_ids)

# Créer le contenu HTML avec un seul tableau récapitulatif
html = f"""
<html>
<head>
  <meta charset="UTF-8">
  <title>Résumé des actionDetails - Tableau Unique</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ background-color: #f2f2f2; padding: 10px; }}
    table {{ border-collapse: collapse; margin-bottom: 30px; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background-color: #f9f9f9; }}
  </style>
</head>
<body>
  <h1>Résumé des actionDetails - Tableau Unique</h1>
  {grouped_df.to_html(index=False, escape=False)}
</body>
</html>
"""

# Sauvegarder le fichier HTML
output_file = "/Users/rezisabashvili/Desktop/summary_single.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Fichier HTML généré avec succès : {output_file}")
