# Handwritten Digit Recognition Using Machine Learning and Neural Networks

A Python project based on the capstone described in the uploaded AI internship report and presentation.

## Project overview

The project classifies isolated handwritten digits from 0 to 9 using the `load_digits` dataset bundled with scikit-learn. Each sample is an 8×8 grayscale image represented by 64 pixel features.

The included pipeline:
1. Loads and explores the dataset.
2. Produces sample-image, class-distribution, and PCA visualizations.
3. Splits the data into an 80% training set and a 20% held-out test set using stratification and `random_state=42`.
4. Compares six classifiers using five-fold stratified cross-validation.
5. Evaluates test-set accuracy, weighted precision, weighted recall, weighted F1, and a confusion matrix.
6. Runs extra sensitivity experiments for SVM settings, k-NN neighbours, and Random Forest tree counts.
7. Saves a fitted best-performing model and charts/reports.

## Technology stack

- Python 3
- NumPy and Pandas
- Matplotlib
- scikit-learn
- joblib
- Jupyter Notebook / Google Colab / VS Code (optional development environments)
- Git and GitHub (version control)

## Algorithms

- Decision Tree
- Logistic Regression
- K-Nearest Neighbours (k=3)
- Random Forest (200 trees)
- Support Vector Machine with RBF kernel (C=10)
- Multi-Layer Perceptron (hidden layers 64 and 32, ReLU, Adam)

Scaling is applied inside pipelines for Logistic Regression, k-NN, SVM, and MLP so that each fold's scaler is fitted only on that fold's training portion. Decision Tree and Random Forest use the raw pixel values.

## Setup

Python 3.10 or later is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

To skip the extra sensitivity experiments:

```bash
python main.py --no-sensitivity
```

## Generated outputs

The program writes results into `outputs/`:

- `sample_digits.png`
- `class_distribution.png`
- `pca_projection.png`
- `model_comparison.csv`
- `model_accuracy_comparison.png`
- `best_model_confusion_matrix.png`
- `best_model_report.txt`
- `prediction_examples.png`
- `sensitivity_experiments.csv`
- `sensitivity_analysis.png`
- `run_summary.json`

The fitted model is written to `models/best_digit_classifier.joblib`.

## Predicting a prepared 8×8 input

Create a NumPy `.npy` file containing an 8×8 numeric array with pixel values on a scale comparable to the dataset (usually 0–16), then run:

```bash
python predict_digit.py sample_digit.npy
```

This helper is intentionally limited to 8×8 arrays. The project does not directly accept camera photos, multi-digit strings, letters, or arbitrary-size images.

## Evaluation notes

The internship report records 98.06% test accuracy for its main RBF-SVM comparison and notes that an unstandardized RBF-SVM reached 99.44% in a separate sensitivity experiment. This implementation runs the experiments again; exact scores can vary with package versions and configuration. Use the generated CSV files as the results for the current run rather than copying the report's scores into outputs.

## Scope and future improvements

Current scope is isolated single digits in the scikit-learn 8×8 dataset. Possible extensions include:
- Training a CNN on MNIST.
- Supporting multi-digit images.
- Adding preprocessing for user-uploaded images.
- Building a small web interface with a human-verification step.
- Performing systematic hyperparameter search with a separate validation strategy.

## References

- scikit-learn documentation: https://scikit-learn.org/
- NumPy documentation: https://numpy.org/doc/
- Pandas documentation: https://pandas.pydata.org/docs/
- Matplotlib documentation: https://matplotlib.org/stable/
