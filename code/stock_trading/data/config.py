import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional

class TickerData(BaseModel):
    ticker: str = Field(..., title="Ticker Symbol")
    prices: Optional[pd.DataFrame] = Field(None, title="Technical Metrics")
    analyst_recs: Optional[pd.DataFrame] = Field(None, title="Analyst Recommendations")
    balance_sheet: Optional[pd.DataFrame] = Field(None, title="Balance Sheet")
    earnings_estimate: Optional[pd.DataFrame] = Field(None, title="Earnings Estimate")

    class Config:
        arbitrary_types_allowed = True