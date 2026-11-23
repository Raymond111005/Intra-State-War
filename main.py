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
LABEL_NAMES = {-1: 'Defender Win', 0: 'Stalemate', 1: 'Attacker Win'}
FEATURE_NAMES = {
    'aeroa': 'Air Superiority', 'surpa': 'Surprise', 'leada': 'Leadership',
    'trnga': 'Training', 'morala': 'Morale', 'logsa': 'Logistics',
    'intela': 'Intelligence', 'techa': 'Technology',
}


def run():
    df = pd.read_csv(DATA_PATH)
    df = df[df['is_hero'] == 1].copy()

    if (df[TARGET] < 0).any():
    # Clean CDB90 file: -1 values are intact, so NaNs are truly unknown outcomes
        df = df.dropna(subset=[TARGET]).copy()
        y = df[TARGET].astype(int)
    else:
        y = df[TARGET].fillna(-1).astype(int)
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=471, stratify=y
    )
    imputer = SimpleImputer(strategy='constant', fill_value=-1, keep_empty_features=True)
    X_train_filled = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_filled = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)

    model = RandomForestClassifier(n_estimators=150, random_state=471,
                                   class_weight={-1: 1.0, 0: 4.0, 1: 1.0}, min_samples_leaf=3)
    model.fit(X_train_filled, y_train)
    predictions = model.predict(X_test_filled)
    baseline = np.full(len(y_test), y_train.mode().iloc[0])
    print(f'Battles with known outcome: {len(df)}')
    print(f'Test accuracy: {accuracy_score(y_test, predictions):.3f}')
    print(f'Test balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}')
    print(f'Majority baseline accuracy: {accuracy_score(y_test, baseline):.3f}')
    print('Confusion matrix (rows=actual, columns=predicted; class order -1, 0, 1):')
    print(confusion_matrix(y_test, predictions, labels=[-1, 0, 1]))

    all_probs = model.predict_proba(X_test_filled)
    max_probs = all_probs.max(axis=1)
    correct_idx = np.where(predictions == y_test.to_numpy())[0]
    wrong_idx = np.where(predictions != y_test.to_numpy())[0]

    cases = []
    if len(correct_idx):
        cases.append(('CASE 1: CONFIDENT CORRECT', correct_idx[np.argmax(max_probs[correct_idx])]))
    if len(wrong_idx):
        cases.append(('CASE 2: LOW-CONFIDENCE MISS', wrong_idx[np.argmin(max_probs[wrong_idx])]))

    explainer = shap.TreeExplainer(model)

    for title, pos in cases:
        example = X_test_filled.iloc[[pos]]
        prediction = int(predictions[pos])
        actual = int(y_test.iloc[pos])
        probs = all_probs[pos]
        prob_str = ', '.join(f'{LABEL_NAMES[c]}: {p:.1%}' for c, p in zip(model.classes_, probs))

        print(f'\n--- {title} ---')
        print(f'Battle    : {df.loc[example.index[0], "name"]} ({df.loc[example.index[0], "war"]})')
        print(f'Predicted : {LABEL_NAMES[prediction]} | Actual: {LABEL_NAMES[actual]}')
        print(f'Confidence: {prob_str}')

        # Explain the predicted class; modern SHAP uses (samples, features, classes).
        explanation = explainer(example)
        class_idx = list(model.classes_).index(prediction)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            contributions = values[0, :, class_idx]
        elif values.ndim == 2:
            contributions = values[0]
        else:
            raise ValueError(f'Unexpected SHAP array shape: {values.shape}')

        ranking = sorted(
            zip(FEATURES, example.iloc[0].to_numpy(), contributions),
            key=lambda item: abs(item[2]),
            reverse=True,
        )
        print('Top SHAP feature contributions (for predicted class):')
        for feature, val, shap_val in ranking[:5]:
            direction = 'FOR' if shap_val >= 0 else 'AGAINST'
            print(f'  {FEATURE_NAMES[feature]:15s} ({feature}={val:>4.1f})  {shap_val:+.4f} [{direction}]')
    print('\nNOTE: This models historical coded outcomes, not future combat performance.')


if __name__ == '__main__':
    run()
