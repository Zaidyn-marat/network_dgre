================================================================
MSR Analysis Pipeline for Open Clusters
================================================================

This repository provides the Python pipeline used to calculate the 
Mass Segregation Ratio (MSR) and perform dynamical age correlation 
analysis.

---
1. CONTENTS
---
- analysis_pipeline.py: The main execution script.
- Cluster_MSR_Dynamical_Data.csv: The merged dataset containing MSR 
  values and dynamical ages (tau) for clusters.
- README.txt: This document.

---
2. GETTING STARTED
---
Dependencies:
Ensure you have the following Python libraries installed:
pip install pandas numpy networkx scipy matplotlib tqdm

Execution steps:
1. Download the original dataset (Hunt et al. 2024) from your Zenodo 
   DOI repository.
2. Ensure the "hunt24.csv" file and the "hunt24/" folder are placed 
   in the same directory as the analysis_pipeline.py script.
3. Run the script:
   python analysis_pipeline.py

---
3. RESULTS
---
- The script generates a merged dataset "NEW_data_for_fig4_plot.csv".
- All scientific figures (PDF format) are saved in the "figures/" 
  directory automatically.

---
4. DATA REFERENCE
---
The results presented are derived from the data hosted at the Zenodo 
repository: [10.5281/zenodo.19051090]