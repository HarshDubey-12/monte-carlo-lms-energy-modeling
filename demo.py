#!/usr/bin/env python3
"""
Demo script for Monte Carlo LMS Energy Modeling.

This script demonstrates the full pipeline: loading data, preprocessing,
training the LMS model, and evaluating prediction uncertainty.
"""

import os
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available, skipping plots.")

from montecarlo_lms import load_data, preprocess_data, LMSRegressor, evaluate_uncertainty, plot_prediction_distribution, plot_prediction_bands, compute_coefficient_of_variation

def main():
    # Set data directory
    data_dir = 'data'

    print("Loading data...")
    train_df, building_df, weather_df = load_data(data_dir)
    print(f"Loaded {len(train_df)} train records, {len(building_df)} building records, {len(weather_df)} weather records.")

    print("Preprocessing data...")
    X_train, y_train, X_val, y_val, FEATURES, TARGET, X_mean, X_std, site_id = preprocess_data(train_df, building_df, weather_df)
    print(f"Selected site {site_id}. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    print("Training LMS model...")
    n_features = X_train.shape[1]
    model = LMSRegressor(n_features=n_features, learning_rate=0.01)
    model.fit(X_train, y_train, n_epochs=5)  # Fewer epochs for demo
    print(f"Training complete. Final loss: {model.loss_history[-1]:.4f}")

    print("Setting up Monte Carlo uncertainty...")
    feature_stds = X_train.std(axis=0)
    weather_features = ['air_temperature', 'dew_temperature', 'wind_speed']
    weather_indices = [FEATURES.index(f) for f in weather_features]
    noise_scales = {idx: 0.05 * feature_stds[idx] for idx in weather_indices}

    print("Evaluating uncertainty on validation batch...")
    N_points = 20
    X_batch = X_val[:N_points]
    y_batch = y_val[:N_points]
    y_mc_pred, y_det_pred, y_mc_mean, y_mc_std, y_mc_lower, y_mc_upper = evaluate_uncertainty(
        model, X_batch, y_batch, noise_scales, n_mc=100  # Fewer samples for demo
    )

    print(f"Mean prediction std (uncertainty): {y_mc_std.mean():.4f}")
    cv = compute_coefficient_of_variation(y_mc_std, y_mc_mean)
    print(f"Demo complete! Check results/figures/ for plots.")

    # Generate and save plots
    if HAS_MATPLOTLIB:
        os.makedirs('results/figures', exist_ok=True)

        plt.figure()
        plot_prediction_distribution(y_mc_pred, y_det_pred, y_mc_mean, idx=0, show=False)
        plt.savefig('results/figures/prediction_distribution.png')
        plt.close()

        plot_prediction_bands(y_det_pred, y_mc_mean, y_mc_lower, y_mc_upper, N_points, show=False)
        plt.savefig('results/figures/prediction_bands.png')
        plt.close()

        print("Plots saved to results/figures/")
    else:
        print("Matplotlib not available, plots not generated.")

if __name__ == "__main__":
    main()