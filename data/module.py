from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from data.config import TickerData

from sklearn.model_selection import train_test_split
from data.utils import (
    collect_sp500_data,
)
from data.feature_engineering import (
    beta,
    weighted_price,
    change_in_recommendations,
    pct_change_signals,
)


class StockDatasetTorch(Dataset):
    def __init__(self, lookback_days: int, batch_size: int):
        self.data = collect_sp500_data(lookback_days, batch_size)
    
    def __post_init__(self):
        # make data aligned by the same date, 
        # use balance sheet dates to gather data by month using all the metrics
        # align data for ML model. This means we need to forward propogate the balance sheet data
        # s.t. we cannot look at the present time to predict the future
        raise NotImplementedError
    
    def _align_dataframes(self):
        raise NotImplementedError
    
    def _forward_fill(self):
        raise NotImplementedError
    
    def _compute_features(self):
        raise NotImplementedError
    
    def _prepare_data(self):
        # per ticker, we need to compute features, forward fill, and align dataframes
        # then, we can prepare train, test, and validation data
        raise NotImplementedError
    
    def prepare_data(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError

class StockDataLoaderTorch(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        
    def get_dataloaders(self, test_size: float = 0.2, val_size: float = 0.1):
        raise NotImplementedError



    