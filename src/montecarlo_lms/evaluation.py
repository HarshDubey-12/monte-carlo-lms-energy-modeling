import numpy as np
import matplotlib.pyplot as plt

def evaluate_uncertainty(model, X_batch, y_batch, noise_scales, n_mc=500):
    """
    Evaluate prediction uncertainty using Monte Carlo simulations.

    Parameters
    ----------
    model : LMSRegressor
        Trained model.
    X_batch : np.ndarray
        Input features batch.
    y_batch : np.ndarray
        True targets batch.
    noise_scales : dict
        Noise scales for perturbation.
    n_mc : int
        Number of Monte Carlo samples.

    Returns
    -------
    y_mc_pred : np.ndarray
        Monte Carlo predictions (n_mc, n_points).
    y_det_pred : np.ndarray
        Deterministic predictions.
    y_mc_mean : np.ndarray
        Mean of MC predictions.
    y_mc_std : np.ndarray
        Std of MC predictions.
    y_mc_lower : np.ndarray
        2.5% percentile.
    y_mc_upper : np.ndarray
        97.5% percentile.
    """
    from .monte_carlo import monte_carlo_perturb

    # Generate MC perturbations
    X_mc = monte_carlo_perturb(X_batch, noise_scales, n_mc=n_mc)

    # MC predictions
    y_mc_pred = np.zeros((n_mc, X_batch.shape[0]))
    for k in range(n_mc):
        y_mc_pred[k] = model.predict(X_mc[k])

    # Deterministic prediction
    y_det_pred = model.predict(X_batch)

    # Uncertainty statistics
    y_mc_mean = y_mc_pred.mean(axis=0)
    y_mc_std = y_mc_pred.std(axis=0)
    y_mc_lower = np.percentile(y_mc_pred, 2.5, axis=0)
    y_mc_upper = np.percentile(y_mc_pred, 97.5, axis=0)

    return y_mc_pred, y_det_pred, y_mc_mean, y_mc_std, y_mc_lower, y_mc_upper

def plot_prediction_distribution(y_mc_pred, y_det_pred, y_mc_mean, idx=0, show=True):
    """
    Plot the prediction distribution for a single sample.
    """
    plt.hist(y_mc_pred[:, idx], bins=40, alpha=0.7)
    plt.axvline(y_det_pred[idx], color="red", linestyle="--", label="Deterministic")
    plt.axvline(y_mc_mean[idx], color="black", linestyle="-", label="MC Mean")
    plt.title("Prediction Distribution (Single Sample)")
    plt.xlabel("Predicted log energy")
    plt.ylabel("Frequency")
    plt.legend()
    if show:
        plt.show()

def plot_prediction_bands(y_det_pred, y_mc_mean, y_mc_lower, y_mc_upper, n_points, show=True):
    """
    Plot prediction bands across multiple points.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_det_pred, label="Deterministic", marker="o")
    plt.plot(y_mc_mean, label="MC Mean", marker="x")
    plt.fill_between(
        range(n_points),
        y_mc_lower,
        y_mc_upper,
        alpha=0.3,
        label="95% Confidence Interval"
    )
    plt.xlabel("Sample Index")
    plt.ylabel("Predicted log energy")
    plt.title("Monte Carlo Uncertainty in LMS Predictions")
    plt.legend()
    if show:
        plt.show()

def compute_coefficient_of_variation(y_mc_std, y_mc_mean):
    """
    Compute coefficient of variation for prediction reliability.
    """
    cv = y_mc_std / (np.abs(y_mc_mean) + 1e-6)
    print("Mean CV:", np.mean(cv))
    print("Max CV:", np.max(cv))
    return cv