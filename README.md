# Monte Carlo LMS Energy Modeling

## Overview

Real-world energy systems operate on noisy, incomplete, and imperfect measurements collected from smart meters and external sources. Treating these inputs as exact values often leads to overconfident models that perform poorly in production.

This project studies how measurement uncertainty affects learning, prediction, and risk in building-level energy consumption modeling using real operational data from commercial buildings.

Key components:

- Monte Carlo simulation to explicitly model measurement uncertainty
- Least Mean Squares (LMS / stochastic gradient descent) to train a linear regression model online/incrementally
- Focus on robustness, convergence behavior, and uncertainty propagation rather than only benchmark accuracy

## Modeling Perspective

The system is modeled as the interaction of:

- **Static building context** — deterministic conditioning information (e.g., building size, usage type)
- **Dynamic operational measurements** — hourly electricity meter readings treated as noisy observations
- **External environmental inputs** — weather station data treated as uncertain exogenous signals

This reflects production monitoring and forecasting systems where ground truth is rarely observable.

## Methodology

- Model measurement uncertainty via Monte Carlo perturbations of observed inputs
- Express predictions as distributions (not single point estimates)
- Train a linear model using LMS/SGD to analyze:
  - stability under noise
  - sensitivity to learning rate
  - convergence behavior in non-stationary settings
- Evaluate using error distributions, tail risk, and robustness metrics (not just averages)

## Why This Matters

In smart buildings, energy optimization platforms, and industrial IoT systems, decisions depend on predictions and on the confidence and reliability of those predictions. This project demonstrates:

- Probabilistic reasoning under uncertainty
- Online learning behavior with noisy data
- Engineering judgment beyond strictly metric-driven modeling

## Scope and Intent

This is a methodology-driven, uncertainty-aware modeling study designed to reflect real-world engineering constraints.

## One-Sentence Takeaway

Accurate predictions are not enough — reliable systems require understanding how uncertainty affects learning and decision-making.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshDubey-12/monte-carlo-lms-energy-modeling.git
   cd monte-carlo-lms-energy-modeling
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Example

```python
import os
from montecarlo_lms import load_data, preprocess_data, LMSRegressor, monte_carlo_perturb, evaluate_uncertainty

# Load data
data_dir = 'data'
train_df, building_df, weather_df = load_data(data_dir)

# Preprocess
X_train, y_train, X_val, y_val, FEATURES, TARGET, X_mean, X_std, site_id = preprocess_data(train_df, building_df, weather_df)

# Train model
n_features = X_train.shape[1]
model = LMSRegressor(n_features=n_features, learning_rate=0.01)
model.fit(X_train, y_train, n_epochs=10)

# Estimate noise scales
feature_stds = X_train.std(axis=0)
weather_features = ['air_temperature', 'dew_temperature', 'wind_speed']
weather_indices = [FEATURES.index(f) for f in weather_features]
noise_scales = {idx: 0.05 * feature_stds[idx] for idx in weather_indices}

# Evaluate uncertainty
X_batch = X_val[:50]
y_batch = y_val[:50]
y_mc_pred, y_det_pred, y_mc_mean, y_mc_std, y_mc_lower, y_mc_upper = evaluate_uncertainty(model, X_batch, y_batch, noise_scales)

print("Mean prediction uncertainty (std):", y_mc_std.mean())
```

### Modules

- `data_loader`: Functions to load datasets.
- `preprocessing`: Data cleaning, feature engineering, and train/val split.
- `monte_carlo`: Monte Carlo perturbation for uncertainty modeling.
- `model`: LMSRegressor class for linear regression.
- `evaluation`: Uncertainty evaluation and plotting.

## Project Structure

```
monte-carlo-lms-energy-modeling/
├── src/
│   └── montecarlo_lms/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── monte_carlo.py
│       ├── model.py
│       └── evaluation.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   └── README.md
├── results/
│   └── figures/
├── tests/
├── README.md
├── requirements.txt
├── .gitignore
└── setup.py
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
