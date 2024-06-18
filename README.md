# Comparative Analysis of Cancer-Specific Biomarkers with NetBio
## Project Description
This is our research project for AI for Bio-Health course, Dept of Life Sciences, POSTECH (2024.03 ~ 2024. 05).   

The goal is to analyze comparatively for biomarkers associated with cancer-specific ICI responses with NetBio framework.
+ Identify cancer-specific biomarker by using NetBio framework
  + NetBio code Implementation, modify for apply for other cancer types
+ Check and Compare biological functions of each cancer-specific pathways
  + Gene Ontology analysis 
  + Pathway Enrichment analysis

Project Silde Link: [Slide](https://docs.google.com/presentation/d/1ytQNVHQPi-oq5dnYlmXozXr5shAuIJ8TXj7mwpdYy1U/edit?usp=sharing)  
Final Report Link: [Report](https://drive.google.com/file/d/1hkF630qSOZRQJzC15KiCEGvkhQawDVUG/view?usp=sharing)
  


## NetBio
Source codes for generating results of "Network-based machine learning approach to predict immunotherapy response in cancer patients". 
We modified NetBio code to apply other cancer data types and get results, important pathways for other cancer types. 
NetBio Paper Link: [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9240063/)

### Requirements
- python (3.6.12)
- pandas (1.1.15)
- numpy (1.19.2)
- scipy (1.5.4)
- matplotlib (3.3.3)
- sklearn (0.24.2)
- lifelines (0.25.7)
- networkx (2.5)
- statsmodels (0.12.2)
- pytorch (1.7.1+cu110)

### Installation
All packages can be installed via pip (https://pypi.org/project/pip/). Generally, a couple of minutes is needed for installing each package.
**e.g.** pip install sklearn

### NetBio predictions
- Code for reproducing leave-one-out cross-validation (LOOCV) and Monte-Carlo cross-validation is provided under the './code/1_within_study_prediction' folder.
- Expected results are provided under './result/1_within_study_prediction' folder.
- The expected run time is under 15 minutes for running LOOCV or Monte-Carlo cross-validation.

