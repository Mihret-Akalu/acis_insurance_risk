# ACIS — Insurance Risk Analytics 

Project: End-to-End Insurance Risk Analytics & Predictive Modeling  


Goals
- EDA to identify low-risk segments and loss-ratio patterns
- Hypothesis testing (province, zipcode, gender)
- Per-zipcode linear models and ML model to suggest premium changes
- Reproducible pipeline with DVC and CI

Repository layout
- data/            # raw data (tracked with dvc)
- notebooks/       # EDA and modeling notebooks
- src/             # scripts & modules
- reports/         # plots and summary
- .github/workflows
- dvc.yaml         # DVC pipeline (future)

How to run
1. install requirements: pip install -r requirements.txt
2. run notebooks in notebooks/
3. use DVC: dvc pull to get data
