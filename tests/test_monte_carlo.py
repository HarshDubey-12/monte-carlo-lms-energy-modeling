import numpy as np
from montecarlo_lms.monte_carlo import monte_carlo_perturb

def test_monte_carlo_perturb():
    np.random.seed(42)
    x = np.random.randn(10, 3)
    noise_scales = {0: 0.1, 2: 0.2}

    X_mc = monte_carlo_perturb(x, noise_scales, n_mc=5)

    assert X_mc.shape == (5, 10, 3)
    # Check that noise was added to specified features
    assert not np.allclose(X_mc[:, :, 0], x[:, 0])
    assert np.allclose(X_mc[:, :, 1], x[:, 1])  # No noise
    assert not np.allclose(X_mc[:, :, 2], x[:, 2])