# Email Spam Classification

Model training and evaluation are complete in BITS Virtual Lab. Streamlit Community Cloud deployment is pending.

## a. Problem statement

Predict whether an email is spam using its word, character and capital-letter counts and frequencies. Compare five classification models on the same test data.

## b. Dataset description

- Dataset: [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase).
- Original data: 4,601 rows and 57 numeric input features.
- Target: `spam`, where 0 means not spam and 1 means spam.
- The downloaded file has no missing values.
- Removed 391 repeated rows, then 6 rows with matching inputs but conflicting labels.
- Data used: 4,204 rows, with 2,528 not-spam and 1,676 spam examples.
- Split: 3,363 training rows and 841 test rows (80:20), keeping similar class proportions. Random seed: 42.
- Logistic Regression, KNN and SVM use scaling fitted only on the training data. The saved models include this scaling.

Source: Hopkins, M., Reeber, E., Forman, G., and Suermondt, J. (1999), *Spambase*, UCI Machine Learning Repository. [DOI](https://doi.org/10.24432/C53G6X). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original data files are kept in `data/`; the notebook applies the cleaning above.

## c. GitHub repository link

[GitHub repository](https://github.com/2026ak05111-source/ml-assignment-2)

## d. Models used

Logistic Regression, Decision Tree, KNN (5 neighbours), Gaussian Naive Bayes and SVM (RBF kernel).

These results are from the BITS Virtual Lab run. All models use the same test rows. Spam is the positive class. AUC uses class probabilities, except for SVM, which uses decision scores.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9239 | 0.9721 | 0.9045 | 0.9045 | 0.9045 | 0.8412 |
| Decision Tree | 0.9227 | 0.9186 | 0.9066 | 0.8985 | 0.9025 | 0.8385 |
| KNN | 0.9168 | 0.9654 | 0.9027 | 0.8866 | 0.8946 | 0.8259 |
| Naive Bayes | 0.8252 | 0.9498 | 0.7035 | 0.9701 | 0.8156 | 0.6881 |
| Support Vector Machine | 0.9310 | 0.9765 | 0.9184 | 0.9075 | 0.9129 | 0.8559 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Precision and recall are both 0.9045. Its F1 score is slightly below SVM and above the other models. |
| Decision Tree | Accuracy is close to Logistic Regression, but its AUC is the lowest. Its recall is slightly lower than Logistic Regression. |
| KNN | Precision is 0.9027, but recall is 0.8866. It misses more spam than the other models in this test. |
| Naive Bayes | Recall is highest at 0.9701, but precision is lowest at 0.7035. It catches most spam but marks more normal emails as spam. |
| Support Vector Machine | It has the highest accuracy, AUC, precision, F1 and MCC. Recall is 0.9075, below Naive Bayes. |
| Overall winner | SVM gives the best overall result on this test set, with the highest F1 (0.9129) and accuracy (0.9310). |

## Run locally

Use Python 3.12. The lab used Python 3.12.7, NumPy 2.5.4, pandas 2.2.3, SciPy 1.18.1, scikit-learn 1.9.1 and joblib 1.4.2. These package versions are listed in `requirements.txt`.

For a fresh setup, run these commands inside `Assignment_2`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install notebook==7.6.5
```

The local environment is already prepared. Activate it and open the notebook:

```bash
source .venv/bin/activate
python -m notebook Assignment_2.ipynb
```

The notebook includes the lab outputs, and the saved models and `test_data.csv` are ready to use. To train again, run all notebook cells. To start the app, use another terminal in the same folder:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

Upload `test_data.csv` and choose a model. The app calculates all six scores and shows a confusion matrix. It does not retrain the models. If an uploaded file contains only one class, AUC is shown as unavailable.

The project uses the notebook results and saved models returned from BITS Virtual Lab. Keep `.venv` and the transfer archives out of GitHub.
