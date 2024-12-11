## Privileged Multitask for Images

IMPORTANT: Download the Eurosat dataset first. (GAP with Highway vs River)

Code files:
- _clean_data.py_. Extraction of RGB (Regular) and NIR (Privileged) bands. Train test split.
- _cv_split.py_. Very first approach with CV.
- _models.py_. Models for the classification, autoencoder and the mult-task approach.
- _utils.py_. Auxiliary functions, loss functions and more.
- _MT_concat.py_. Main code for a MT approach where predictedNIR images are expanded to 3 bands and concatenated below the RGB images for the classification task. Models learned: Upper, Lower, KT, PFD, TPD, MT, MT-PFD, MT-TPD.
- _MT_bands.py_. Main code for a MT approach where predicted NIR images are added as a new band to the RGB images for the classification task. Models learned: Upper, Lower, KT, PFD, TPD, MT, MT-PFD, MT-TPD.

Datasets:
- Eurosat: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset
