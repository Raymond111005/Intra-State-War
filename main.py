"""CS471 checkpoint: historical battle outcome classification and SHAP explanations.

Run: python main.py
Dataset: battles.csv in the same directory.
IMPORTANT: wina coding and feature timing require verification against the codebook.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).with_name('battles.csv')
FEATURES = ['aeroa', 'surpa', 'leada', 'trnga', 'morala', 'logsa', 'intela', 'techa']
TARGET = 'wina'


def run():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET]).copy()
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=471, stratify=y
    )
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    X_train_filled = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)

    model = RandomForestClassifier(n_estimators=150, random_state=471,
                                   class_weight='balanced', min_samples_leaf=3)
    model.fit(X_train_filled, y_train)
    predictions = model.predict(X_test_filled)
    baseline = np.full(len(y_test), y_train.mode().iloc[0])
    print(f'Battles with known outcome: {len(df)}')
    print(f'Test accuracy: {accuracy_score(y_test, predictions):.3f}')
    print(f'Test balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}')
    print(f'Majority baseline accuracy: {accuracy_score(y_test, baseline):.3f}')
    print('Confusion matrix (rows=actual, columns=predicted; class order 0,1):')
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    example = X_test_filled.iloc[[0]]
    prediction = int(model.predict(example)[0])
    probs = model.predict_proba(example)[0]
    print(f'\nExample battle: {df.loc[example.index[0], "name"]}')
    print(f'Predicted wina code: {prediction} (actual: {int(y_test.loc[example.index[0]])})')
    print('Predicted class probabilities:', dict(zip(model.classes_.tolist(), probs.round(3).tolist())))

    # Explain the predicted class; modern SHAP uses (samples, features, classes).
    explanation = shap.TreeExplainer(model)(example)
    class_idx = list(model.classes_).index(prediction)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        contributions = values[0, :, class_idx]
    elif values.ndim == 2:
        contributions = values[0]
    else:
        raise ValueError(f'Unexpected SHAP array shape: {values.shape}')
    ranking = sorted(zip(FEATURES, contributions), key=lambda item: abs(item[1]), reverse=True)
    print('\nTop SHAP feature contributions (for predicted class):')
    for feature, value in ranking[:5]:
        print(f'  {feature:10s} {value:+.4f}')
    print('\nNOTE: This models historical coded outcomes, not future combat performance.')


if __name__ == '__main__':
    run()
