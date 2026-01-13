import pandas as pd
import os

def load_data(data_dir):
    """
    Load the training, building metadata, and weather datasets.

    Parameters
    ----------
    data_dir : str
        Path to the data directory.

    Returns
    -------
    train_df : pd.DataFrame
        Training data.
    building_df : pd.DataFrame
        Building metadata.
    weather_df : pd.DataFrame
        Weather data.
    """
    train_path = os.path.join(data_dir, 'train.csv')
    building_path = os.path.join(data_dir, 'building_metadata.csv')
    weather_path = os.path.join(data_dir, 'weather_train.csv')

    train_df = pd.read_csv(train_path)
    building_df = pd.read_csv(building_path)
    weather_df = pd.read_csv(weather_path)

    return train_df, building_df, weather_df