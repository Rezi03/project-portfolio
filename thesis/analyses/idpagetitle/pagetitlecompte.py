import pandas as pd

# Charger le fichier CSV
file_path = '~/Desktop/projet_la/bdd/bdd.csv'
df = pd.read_csv(file_path, sep=';', encoding='UTF-8-SIG', low_memory=False)

# Dictionnaire pour stocker le comptage de chaque pageTitle par actionDetails
counts_dict = {}

# Parcourir les 218 colonnes de pageTitle
for n in range(218):
    col_title = f"pageTitle (actionDetails {n})"
    if col_title in df.columns:
        # Compter les occurrences non nulles de chaque pageTitle dans cette colonne
        counts = df[col_title].dropna().value_counts()
        counts.name = f"actionDetails_{n}"
        counts_dict[f"actionDetails_{n}"] = counts

# Combiner les séries en un DataFrame unique, indexé par pageTitle
combined_counts = pd.concat(counts_dict.values(), axis=1, sort=False).fillna(0).astype(int)

# Créer une colonne 'Total' qui somme les occurrences sur toutes les colonnes
combined_counts['Total'] = combined_counts.sum(axis=1)

# Réinitialiser l'index pour que 'pageTitle' devienne une colonne
combined_counts = combined_counts.reset_index().rename(columns={'index': 'pageTitle'})

# Créer le contenu HTML final avec le tableau récapitulatif
html = f"""
<html>
<head>
  <meta charset="UTF-8">
  <title>Comptage des pageTitle par actionDetails</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
    }}
    h1 {{
      text-align: center;
      background-color: #f2f2f2;
      padding: 10px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 20px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
    }}
    th {{
      background-color: #f9f9f9;
    }}
  </style>
</head>
<body>
  <h1>Comptage des pageTitle par actionDetails</h1>
  {combined_counts.to_html(index=False, escape=False)}
</body>
</html>
"""

# Sauvegarder le fichier HTML
output_file = "/Users/rezisabashvili/Desktop/combined_counts.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Fichier HTML généré avec succès : {output_file}")
