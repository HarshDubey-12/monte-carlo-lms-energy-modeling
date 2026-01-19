import numpy as np
from montecarlo_lms.model import LMSRegressor

def test_lms_regressor():
    # Simple test
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X @ np.array([1.5, -2.0]) + 0.1 * np.random.randn(100)

    model = LMSRegressor(n_features=2, learning_rate=0.01)
    model.fit(X, y, n_epochs=10)

    assert len(model.loss_history) == 10
    assert model.loss_history[-1] < model.loss_history[0]  # Loss should decrease

    pred = model.predict(X[:5])
    assert pred.shape == (5,)