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

## API Documentation

This section provides detailed documentation for each module in the `montecarlo_lms` package.

### `data_loader` Module

#### `load_data(data_dir)`
Loads the training, building metadata, and weather datasets from CSV files.

**Parameters:**
- `data_dir` (str): Path to the directory containing the CSV files (`train.csv`, `building_metadata.csv`, `weather_train.csv`).

**Returns:**
- `train_df` (pd.DataFrame): Training data with columns like `building_id`, `meter`, `timestamp`, `meter_reading`.
- `building_df` (pd.DataFrame): Building metadata with columns like `building_id`, `site_id`, `square_feet`, `year_built`, `primary_use`.
- `weather_df` (pd.DataFrame): Weather data with columns like `site_id`, `timestamp`, `air_temperature`, `dew_temperature`, `wind_speed`.

### `preprocessing` Module

#### `preprocess_data(train_df, building_df, weather_df, selected_site_id=None)`
Preprocesses the data by selecting a site, merging datasets, handling missing values, feature engineering, and splitting into train/validation sets.

**Parameters:**
- `train_df` (pd.DataFrame): Raw training data.
- `building_df` (pd.DataFrame): Raw building metadata.
- `weather_df` (pd.DataFrame): Raw weather data.
- `selected_site_id` (int, optional): Pre-selected site ID. If None, selects automatically based on data coverage.

**Returns:**
- `X_train_norm` (np.ndarray): Normalized training features (n_samples, n_features).
- `y_train` (np.ndarray): Training targets (log-transformed meter readings).
- `X_val_norm` (np.ndarray): Normalized validation features.
- `y_val` (np.ndarray): Validation targets.
- `FEATURES` (list): List of feature names (e.g., ['air_temperature', 'dew_temperature', 'wind_speed', 'hour', 'dayofweek', 'log_square_feet']).
- `TARGET` (str): Target name ('log_meter_reading').
- `X_mean` (np.ndarray): Mean values for feature normalization.
- `X_std` (np.ndarray): Standard deviation values for feature normalization.
- `site_id` (int): Selected site ID.

**Key Variables:**
- `MIN_TRAIN_RECORDS`, `MIN_WEATHER_RECORDS`: Thresholds for site selection based on data availability.
- `site_summary_filtered`: Filtered sites meeting coverage criteria.
- `balanced_score`: Metric for site selection (lower is better balance between train and weather records).
- `data_df`: Merged and cleaned DataFrame.
- `FEATURES`: Engineered features including time-based (hour, dayofweek) and transformed (log_square_feet).
- `X_train_norm`, `X_val_norm`: Z-score normalized features using training set statistics.

### `monte_carlo` Module

#### `monte_carlo_perturb(x, noise_scales, n_mc=100)`
Generates Monte Carlo perturbed versions of input features by adding noise to specified dimensions.

**Parameters:**
- `x` (np.ndarray): Normalized input features (n_points, n_features).
- `noise_scales` (dict): Mapping of feature_index (int) to noise standard deviation (float).
- `n_mc` (int): Number of Monte Carlo samples (default: 100).

**Returns:**
- `X_mc` (np.ndarray): Perturbed features (n_mc, n_points, n_features).

**Key Variables:**
- `n_points`, `n_features`: Dimensions of input data.
- `X_mc`: Array to store perturbed samples.
- `noise`: Random noise drawn from normal distribution for each feature and sample.

### `model` Module

#### Class `LMSRegressor`
Implements Least Mean Squares (LMS) regression using stochastic gradient descent.

**Attributes:**
- `lr` (float): Learning rate.
- `w` (np.ndarray): Weight vector (n_features,).
- `b` (float): Bias term.
- `loss_history` (list): List of epoch losses for diagnostics.

**Methods:**

##### `__init__(self, n_features, learning_rate=0.01)`
Initializes the model.

**Parameters:**
- `n_features` (int): Number of input features.
- `learning_rate` (float): Step size for updates (default: 0.01).

##### `predict(self, X)`
Predicts outputs for given inputs.

**Parameters:**
- `X` (np.ndarray): Input features (n_samples, n_features).

**Returns:**
- `predictions` (np.ndarray): Predicted values (n_samples,).

##### `update(self, x, y)`
Performs one LMS update on a single sample.

**Parameters:**
- `x` (np.ndarray): Single input sample (n_features,).
- `y` (float): True target value.

**Returns:**
- `error` (float): Prediction error.

##### `fit(self, X, y, n_epochs, shuffle=True)`
Trains the model for multiple epochs.

**Parameters:**
- `X` (np.ndarray): Training features (n_samples, n_features).
- `y` (np.ndarray): Training targets (n_samples,).
- `n_epochs` (int): Number of training epochs.
- `shuffle` (bool): Whether to shuffle data each epoch (default: True).

### `evaluation` Module

#### `evaluate_uncertainty(model, X_batch, y_batch, noise_scales, n_mc=500)`
Evaluates prediction uncertainty using Monte Carlo simulations.

**Parameters:**
- `model` (LMSRegressor): Trained model.
- `X_batch` (np.ndarray): Batch of input features.
- `y_batch` (np.ndarray): Batch of true targets.
- `noise_scales` (dict): Noise scales for perturbation.
- `n_mc` (int): Number of Monte Carlo samples (default: 500).

**Returns:**
- `y_mc_pred` (np.ndarray): Monte Carlo predictions (n_mc, n_points).
- `y_det_pred` (np.ndarray): Deterministic predictions.
- `y_mc_mean` (np.ndarray): Mean of MC predictions.
- `y_mc_std` (np.ndarray): Std of MC predictions.
- `y_mc_lower` (np.ndarray): 2.5% percentile (lower CI).
- `y_mc_upper` (np.ndarray): 97.5% percentile (upper CI).

**Key Variables:**
- `N_points`: Number of samples in batch.
- `cv`: Coefficient of variation (y_mc_std / |y_mc_mean|), measures relative uncertainty.

#### `plot_prediction_distribution(y_mc_pred, y_det_pred, y_mc_mean, idx=0, show=True)`
Plots the prediction distribution histogram for a single sample.

**Parameters:**
- `y_mc_pred`, `y_det_pred`, `y_mc_mean`: Prediction arrays.
- `idx` (int): Sample index (default: 0).
- `show` (bool): Whether to display the plot (default: True).

#### `plot_prediction_bands(y_det_pred, y_mc_mean, y_mc_lower, y_mc_upper, n_points, show=True)`
Plots prediction bands with confidence intervals.

**Parameters:**
- `y_det_pred`, `y_mc_mean`, `y_mc_lower`, `y_mc_upper`: Prediction arrays.
- `n_points` (int): Number of points.
- `show` (bool): Whether to display the plot (default: True).

#### `compute_coefficient_of_variation(y_mc_std, y_mc_mean)`
Computes and prints coefficient of variation statistics.

**Parameters:**
- `y_mc_std`, `y_mc_mean` (np.ndarray): Std and mean predictions.

**Returns:**
- `cv` (np.ndarray): Coefficient of variation values.
