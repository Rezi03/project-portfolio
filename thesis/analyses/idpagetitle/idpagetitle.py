import pandas as pd


file_path = '~/Desktop/projet_la/bdd/bdd.csv'

df = pd.read_csv(file_path, sep=';', encoding='UTF-8-SIG', low_memory=False)

# Début du contenu HTML
html = """
<html>
<head>
  <meta charset="UTF-8">
  <title>Résumé des actionDetails</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h2 { background-color: #f2f2f2; padding: 10px; }
    table { border-collapse: collapse; margin-bottom: 30px; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #f9f9f9; }
  </style>
</head>
<body>
  <h1>Résumé des actionDetails</h1>
"""

# Parcourir les 218 paires de colonnes
for n in range(218):
    col_title = f"pageTitle (actionDetails {n})"
    col_id = f"pageIdAction (actionDetails {n})"
    
    html += f"<h2>actionDetails {n}</h2>"
    if col_title in df.columns and col_id in df.columns:
        # Extraire et nettoyer les données pour cette paire
        temp_df = df[[col_title, col_id]].dropna(subset=[col_title])
        temp_df = temp_df.rename(columns={col_title: "pageTitle", col_id: "pageIdAction"})
        # Supprimer les doublons sur "pageTitle"
        temp_df = temp_df.drop_duplicates(subset=["pageTitle"])
        # Convertir le DataFrame en table HTML
        html += temp_df.to_html(index=False, escape=False)
    else:
        html += "<p>Les colonnes n'existent pas dans ce segment.</p>"

# FinHTML
html += """
</body>
</html>
"""

# Sauvegarder fichier HTML
output_file = "/Users/rezisabashvili/Desktop/summary.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Fichier HTML généré : {output_file}")