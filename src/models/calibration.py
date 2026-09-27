import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence) * prop_in_bin

    return float(ece)


def calibrate_model(model, X_train, y_train, method='sigmoid'):
    """
    Calibrate a model using Platt scaling ('sigmoid') or Isotonic regression ('isotonic').
    """
    calibrated_model = CalibratedClassifierCV(estimator=model, method=method, cv=2)
    calibrated_model.fit(X_train, y_train)
    return calibrated_model


def map_risk_tier(prob):
    """
    Map a probability to the 4-Tier Risk Policy.

    Boundaries follow the trained model's own classes on the 0-100 risk
    index (LOW < 35, WATCH 35-65, HIGH >= 65) after normalising to [0, 1].
    The WATCH class maps to the Moderate tier; the LOW class is split into
    No Risk (< 0.05, essentially a healthy baseline) and Low Risk.
    """
    if prob < 0.05:
        return "No Risk", "Routine monitoring and standard milking hygiene protocols."
    elif prob < 0.35:
        return "Low Risk", "Mild abnormality or rising trend; observe cow and repeat sensor measurement at next milking."
    elif prob < 0.65:
        return "Moderate Risk", "Actionable risk; inspect cow/milking process, perform California Mastitis Test (CMT), notify farm supervisor."
    else:
        return "High Risk", "Urgent veterinary review, confirmatory SCC/CMT/lab testing, isolate milking equipment workflow."


def evaluate_calibration(y_true, y_prob_raw, y_prob_platt, y_prob_iso):
    """Compare uncalibrated, Platt, and Isotonic probabilities."""
    results = {
        'Uncalibrated': {
            'Brier Score': brier_score_loss(y_true, y_prob_raw),
            'ECE': calculate_ece(y_true, y_prob_raw)
        },
        'Platt Scaled (Sigmoid)': {
            'Brier Score': brier_score_loss(y_true, y_prob_platt),
            'ECE': calculate_ece(y_true, y_prob_platt)
        },
        'Isotonic Calibrated': {
            'Brier Score': brier_score_loss(y_true, y_prob_iso),
            'ECE': calculate_ece(y_true, y_prob_iso)
        }
    }
    return results
