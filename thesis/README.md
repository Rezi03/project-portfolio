# Master's Thesis - Learning Analytics

This repository contains the complete materials for my Master's thesis in Learning Analytics (M1 level).

## Repository Structure
```
.
├── final_submission
│ ├── thesis_final_document.pdf # Final version of the thesis document
│ ├── thesis_presentation.pdf # Oral presentation slides of the thesis
│ └── thesis_scoringsheet.pdf # Document showing the scoring/notes for the thesis
│
├── docs
│ ├── cer # Ethics Committee documents for URCA submission
│ │ └── CER_submission_document.pdf
│ ├── engagement # Articles related to Learning Analytics and Engagement Analytics
│ │ ├── article1.pdf
│ │ └── article2.pdf
│ └── profils # Characterization of Learning Datasets and related presentation
│ ├── new_way_characterize_article.pdf
│ └── oral_presentation.pdf
│
├── data
│ ├── educapacite_data.csv # Anonymized dataset of traces left on the Educapacite website (CSV)
│ └── educapacite_data.xlsx # Same dataset in Excel format
│
├── digital_traces
│ ├── data
│ │ ├── anonymedata.csv # Anonymized test trace dataset
│ │ ├── data.xlsx # Trace dataset for hypothesis testing
│ │ └── rezi.xlsx # Additional trace data
│ └── analyses.ipynb # Python notebook with analysis and hypothesis verification
│
└── analyses
├── first_analyses
│ ├── test.R # Initial descriptive statistical analysis
│ └── description # Summary and synthesis of the first analyses
│
├── anonymize
│ ├── anonymiser_both.ipynb # Notebook anonymizing both email and visit IP columns
│ ├── anonymiser_email.ipynb # Notebook anonymizing only email column
│ ├── anonymiser_visitIp.ipynb # Notebook anonymizing only visit IP column
│ └── bdd_test.csv # Sample anonymized database to test and guarantee anonymization
│
├── type
│ ├── concordance.html # Analysis results on typical paths
│ ├── path.html
│ └── path.ipynb
│
├── indicators
│ ├── test_indicators.ipynb # Notebook for indicator tests
│ ├── other
│ │ ├── analyses.pdf # Documents about indicators and their calculations (by Mr. Prosper Sanou)
│ │ └── indicators.pdf
│ └── description
│ ├── columns.R # Script extracting unique columns
│ └── description.pdf # Definitions and descriptions of all columns
│
├── clustering
│ └── clustering.ipynb # Clustering analysis notebook
│
└── id_page_title
├── idpagetitelunique.py # Script analyzing page ID uniqueness
├── idpagetitle.html # Output HTML with results
├── idpagetitle.py # Script analyzing page titles
├── idpagetitleunique.html # Output HTML for unique titles
├── pagetitlecompte.html # HTML results for page title counts
└── pagetitlecompte.py # Script for counting page titles
```

## How to navigate

- Start with the `final_submission` folder to see the final thesis document, presentation, and scoring sheet.
- Check the `docs` folder for supporting articles and ethics submission documentation.
- The `data` folder contains the anonymized datasets used for analysis.
- The `digital_traces` folder holds datasets created for hypothesis testing and the Python notebook for analysis.
- The main analyses are in the `analyses` folder, organized by themes (descriptive, anonymization, path analysis, indicators, clustering, and page title analysis).

---

Feel free to explore each folder to understand the progression from raw data to final results and documents.

---

If you have any questions or want to discuss the thesis or the analyses, feel free to contact me.

---

**Rezi Sabashvili**  
Master's Student in Learning Analytics  
University of Reims Champagne-Ardenne  
rezisabashvili1@gmail.com