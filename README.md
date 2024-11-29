## Privileged Multitask for Images


Code files:
- _clean_data.py_. Extraction of RGB (Regular) and NIR (Privileged) bands. Train test split.
- _cv_split.py_. Very first approach with CV.
- _main.py_. Initial code to compute the LOWER (Regular) and UPPER (Privileged + Regular) model. Objective: Error rate UPPER > Error rate LOWER
- _utils.py_. Function for the extraction of RGB (Regular) and NIR (Privileged) bands.


Datasets:
- Eurosat: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
