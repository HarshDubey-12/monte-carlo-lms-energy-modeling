import pytest
import pandas as pd
import os
from montecarlo_lms.data_loader import load_data

def test_load_data():
    # Assuming data files exist
    data_dir = 'data'
    if os.path.exists(os.path.join(data_dir, 'train.csv')):
        train_df, building_df, weather_df = load_data(data_dir)
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(building_df, pd.DataFrame)
        assert isinstance(weather_df, pd.DataFrame)
        assert len(train_df) > 0
        assert len(building_df) > 0
        assert len(weather_df) > 0
    else:
        pytest.skip("Data files not found")