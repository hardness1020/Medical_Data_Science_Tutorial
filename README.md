<h1 align="center">Medical Data Science Tutorial</h1>

<div align="center">

</div>

## 📝 Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Content](#content)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

</br>

## 🧐 About <a name = "about"></a>

### Aim

A tutorial for establishing an ensemble model to stratify patients with acute myeloid leukemia (AML) into three risk groups based on non-clinician-initiated data. 

The performance of the model is compared to the [European LeukemiaNet (ELN) risk stratification system](https://ashpublications.org/blood/article/129/4/424/36196/Diagnosis-and-management-of-AML-in-adults-2017-ELN).

### Dataset

The tabular data used in this tutorial is from the paper: [Unified classification and risk-stratification in Acute Myeloid Leukemia](https://www.nature.com/articles/s41467-022-32103-8#MOESM1).

- NCRI cohort (for training and validation)
- SG cohort (for testing, external cohort).

<!-- The tutorial is based on the paper by [Haferlach et al. (2018)](https://www.nature.com/articles/s41591-018-0026-7).  -->



</br>

## 🏁 Getting Started <a name = "getting_started"></a>

**If you don't have GPU, try using [Google Colab](https://colab.research.google.com/github/hardness1020/Medical_Data_Science_Tutorial/blob/main/tutorial.ipynb).**

### Installing (local usage)

Install the packages

```
pip3 install -r requirements.txt
```

</br>

## 🔧 Content <a name = "content"></a>

The tutorial is divided into three parts:

### 1. Data Preprocessing

- Feature selection
- Data normalization

### 2. Model training

- Model selection: random forest, xgboost
- Hyperparameter optimizer: hyperopt
- Ensemble: loss-based weighting

### 3. Model evaluation

- Performance metrics: Accuracy, F1-score
- Visualization: Confusion matrix
- Survival Analysis: Kaplan-Meier estimator, C-index

</br>

## 🎈 Usage <a name="usage"></a>

### Folder Structure

    ├── dataset : row data
    │   ├── NCRI.tsv : NCRI cohort (for training and validation)
    │   └── SG.tsv   : SG cohort (for testing, external cohort)
    │
    ├── utils: utility functions
    │   ├── hyperparameters
    │   │   ├── hyperoptimizer.py : hyperparameter optimizer
    │   │   └── space.py          : hyperparameters spaces for each model
    |   ├── aml_spliter.py          : split and normalize the dataset into training, validation set 
    |   ├── get_image_bytes.py      : plot and convert confusion matrix to bytes
    |   ├── KM_survival_analysis.py : plot the survival curve and calculate the p-value
    |   └── selected_features.py    : feature selected in the study
    |   
    ├── tutorial.ipynb: tutorial (not yet provided)
    |
    (The tutorial will produce the following files)
    |
    ├── data_preprocessed : preprocessed data
    │   ├── NCRI.csv
    │   └── SG.csv
    |
    └── ESB_result
        ├── best_trial : store the best parameters of each model
        ├── models     : store each model with the best parameters and weights of each model
        └── prediction : store the predictions of each model
            ├── train      : store the predictions of the training set
            ├── validation : store the predictions of the validation set
            └── external   : store the predictions of the test set
    
    
    

### Tutorial

Follow the step by step in **tutorial.ipynb**


</br>

## ✍️ Authors <a name = "authors"></a>

- [@hardness1020](https://github.com/hardness1020)

</br>

## 🎉 Reference <a name = "reference"></a>

