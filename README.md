## Privileged Multitask for Images

IMPORTANT: Download the Eurosat dataset first. (GAP with Highway vs River)

Code files:
- _clean_data.py_. Extraction of RGB (Regular) and NIR (Privileged) bands. Train test split.
- _cv_split.py_. Very first approach with CV.
- _bounds.py_. Initial code to compute the LOWER (Regular) and UPPER (Privileged + Regular) model. Objective: Error rate UPPER > Error rate LOWER
- _utils.py_. Function for the extraction of RGB (Regular) and NIR (Privileged) bands.
- _AE_C.py_. Autoencoder and classifier.
- _MT.py_. Multitask. Joint training AE and classifier.

Datasets:
- Eurosat: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
