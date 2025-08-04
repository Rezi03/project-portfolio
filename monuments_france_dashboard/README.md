# Master's Project - Interactive Analysis of French Monuments

This repository contains the complete materials for our Master's group project (M1 level) focused on building an interactive Python application to explore and analyze French monuments based on a structured dataset.

## Repository Structure

```
📁 app
│ ├── 📄 main.py                # Main script to run the application
│ ├── 📄 queries.py             # Functions to handle user queries
│ ├── 📄 data_processing.py     # Data loading and cleaning utilities
│ └── 📄 utils.py               # Miscellaneous helper functions
│
📁 data
│ ├── 📄 monuments.csv          # Main dataset in CSV format
│ └── 📄 monuments.xl           # Main dataset in Excel format
│
📁 docs
│ ├── 📄 final_report.pdf       # Final written report
│ ├── 📄 roadmap.pdf            # Project roadmap and milestones
│ ├── 📄 flowchart.png          # Diagram showing system workflow
│ └── 📄 presentation.pptx      # Project presentation slides
│
📁 tests
│ ├── 📄 interact_test.py       # Tests for interactive features
│ ├── 📄 bivariate_test.py      # Tests for bivariate analysis
│ └── 📄 data_integrity_test.py # Dataset quality checks
│
📁 tools
│ └── 📄 constants.py           # Global constants and configuration
│
📄 requirements.txt # Python dependencies
📄 README.md # Project overview and documentation
```


## How to navigate

- Start with the `app` folder to see the main application code and how user queries are handled.
- The `data` folder contains the monument datasets in both `.csv` and `.xlsx` formats.
- All project documentation is in the `docs` folder, including the final report, presentation, roadmap, and a process flowchart.
- The `tests` folder includes scripts to test core functionalities and ensure data consistency.
- The `tools` folder contains constants and reusable config elements used throughout the app.

## ▶How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/french-monuments-query-app.git
   cd french-monuments-query-app

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the main application:
    ```bash
    python app/main.py
  
---

Feel free to explore each section to understand how the interactive tool was built and how we structured our code, data, and analysis pipeline.

---

If you have any questions or want to learn more about this project, feel free to get in touch.

---

**Rezi Sabashvili**  
Master's Student in Learning Analytics  
University of Reims Champagne-Ardenne  
rezisabashvili1@gmail.com