# Project Report: Handwritten Digit Recognition Using Machine Learning and Neural Networks

## 1. Abstract

This project implements and compares six machine-learning classifiers for recognising isolated handwritten digits from 0 to 9. The dataset bundled with scikit-learn contains 1,797 grayscale images of size 8×8, represented as 64 numerical pixel features. Python, NumPy, Pandas, Matplotlib, and scikit-learn are used for data analysis, modelling, evaluation, and visualisation. The dataset is split using a stratified 80:20 train-test split, and five-fold stratified cross-validation is used for model comparison. The program reports accuracy, weighted precision, weighted recall, weighted F1-score, and a confusion matrix, and saves the best model selected by held-out test accuracy. Results are generated when the program is run and should be taken from `outputs/model_comparison.csv`.

## 2. Problem Statement

Handwritten digits differ in shape, slant, stroke width, and writing style. Similar-looking digits can be difficult to distinguish using fixed rules. The aim is to train and compare machine-learning models that classify grayscale digit images and identify a strong model for this dataset.

## 3. Objectives

- Explore the dataset and visualise its class distribution.
- Prepare the 64 pixel features for classification.
- Train six classification algorithms.
- Use stratified splitting and five-fold cross-validation.
- Compare the models using multiple evaluation metrics.
- Inspect errors through a confusion matrix and example predictions.
- Run additional experiments on selected hyperparameters.
- Save the fitted model for reuse.

## 4. Dataset

| Property | Description |
|---|---|
| Dataset | `sklearn.datasets.load_digits` |
| Number of samples | 1,797 |
| Image dimensions | 8×8 grayscale |
| Features per sample | 64 |
| Classes | 10 (digits 0–9) |
| Pixel-value range | 0–16 |
| Training split | 80% |
| Test split | 20% |
| Split method | Stratified, random state 42 |

## 5. Technology Stack

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical arrays and transformations |
| Pandas | Tabular summaries and CSV result files |
| Matplotlib | Dataset plots and model evaluation charts |
| scikit-learn | Dataset, preprocessing, classifiers, cross-validation and metrics |
| joblib | Saving/loading the fitted model |
| Jupyter / Colab / VS Code | Optional development environments |
| Git / GitHub | Version control and code sharing |

## 6. Methodology

1. Load the dataset and inspect sample images, class counts, and feature dimensions.
2. Flatten each 8×8 image into 64 pixel features.
3. Create a stratified 80:20 training/test split.
4. Train six classifiers on the training data.
5. Evaluate training-fold performance with five-fold stratified cross-validation.
6. Evaluate each fitted model once on the held-out test data.
7. Compare test accuracy, weighted precision, weighted recall, and weighted F1-score.
8. Visualise model accuracy and the best model's confusion matrix.
9. Run sensitivity experiments for SVM settings, k-NN neighbour counts, and Random Forest tree counts.
10. Save results and the best model.

## 7. Models

- Decision Tree: baseline tree-based classifier.
- Logistic Regression: linear classification baseline with scaled features.
- K-Nearest Neighbours: neighbourhood-based classifier with k=3.
- Random Forest: ensemble of 200 decision trees.
- SVM (RBF): nonlinear margin-based classifier with an RBF kernel and C=10.
- MLP Neural Network: feed-forward neural network with hidden layers of 64 and 32 units, ReLU activation, and Adam optimisation.

## 8. Evaluation

The main metrics are accuracy, weighted precision, weighted recall, and weighted F1-score. A confusion matrix shows the number of correct and incorrect predictions for each digit class. Cross-validation is used to estimate performance on the training portion; the held-out test split is used for final comparison.

**Insert actual run results here:** refer to `outputs/model_comparison.csv`, `outputs/best_model_report.txt`, and `outputs/run_summary.json`. Scores can vary across software versions and experiment settings.

The supplied internship report records 98.06% test accuracy for its main RBF-SVM comparison. It also records 99.44% for an unstandardized SVM in a separate sensitivity experiment. Those values are contextual reference results, not hard-coded outputs of this implementation.

## 9. Outputs

Running `main.py` creates dataset visualisations, model comparison charts and CSV files, a confusion matrix, sample predictions, sensitivity-analysis results, a run summary, and a saved model.

## 10. Limitations

- The dataset uses small 8×8 grayscale images.
- The project is limited to single digits, not multi-digit numbers or letters.
- Arbitrary camera images are not accepted by the prediction helper.
- Results on this controlled benchmark should not be assumed to transfer directly to real-world handwriting.

## 11. Future Scope

- Train a CNN on MNIST.
- Add preprocessing for uploaded images.
- Extend recognition to multi-digit input.
- Build a web application for interactive predictions.
- Add human verification and systematic hyperparameter tuning.

## 12. Conclusion

The project demonstrates an end-to-end supervised machine-learning workflow, from dataset exploration and preprocessing to model comparison and evaluation. It compares classical classifiers and an MLP neural network under a common split and reports multiple metrics instead of relying on accuracy alone. The best-performing model for the current run is saved for later use.
