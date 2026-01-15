import numpy as np

def monte_carlo_perturb(x, noise_scales, n_mc=100):
    """
    Generate Monte Carlo perturbed versions of input data.

    Parameters
    ----------
    x : np.ndarray
        Normalized input features (n_points, n_features)
    noise_scales : dict
        Mapping feature_index -> noise std
    n_mc : int
        Number of Monte Carlo samples

    Returns
    -------
    X_mc : np.ndarray
        Shape (n_mc, n_points, n_features)
    """
    n_points, n_features = x.shape

    # Allocate MC array safely
    X_mc = np.repeat(x[None, :, :], n_mc, axis=0)

    for idx, scale in noise_scales.items():
        noise = np.random.normal(
            loc=0.0,
            scale=scale,
            size=(n_mc, n_points)
        )
        X_mc[:, :, idx] += noise

    return X_mc