import logging
import pandas as pd
from typing import Optional, Dict

def fetch_nvdr_data() -> Optional[pd.DataFrame]:
    """
    Scrapes NVDR trading data from SET website.
    Since SET blocks basic requests (403), this uses a placeholder for Selenium/Playwright 
    or a third-party API.
    """
    logging.info("Fetching NVDR data (simulated/placeholder)...")
    try:
        # In a real production scenario, you would use Selenium to bypass Incapsula
        # Example using selenium (requires 'pip install selenium webdriver-manager'):
        # from selenium import webdriver
        # options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        # driver = webdriver.Chrome(options=options)
        # driver.get('https://www.set.or.th/api/set/nvdr/stat/trading')
        # json_data = driver.find_element(By.TAG_NAME, 'body').text
        # return pd.read_json(json_data)
        
        # Placeholder empty dataframe to prevent breaking the pipeline
        df = pd.DataFrame(columns=['symbol', 'nvdr_net_value', 'foreign_net_value'])
        return df
    except Exception as e:
        logging.error(f"Failed to fetch NVDR data: {e}")
        return None

def get_macro_features(date: str) -> Dict[str, float]:
    """
    Fetches macroeconomic features for a specific date using Yahoo Finance.
    Returns dictionary with THB_USD and US_10Y_YIELD.
    """
    import yfinance as yf
    logging.info(f"Fetching Macro features for {date}...")
    try:
        # Fetch USD/THB exchange rate
        thb_data = yf.download("THB=X", start=date, progress=False)
        thb_val = float(thb_data['Close'].iloc[0].item()) if not thb_data.empty else 0.0
        
        # Fetch US 10-Year Treasury Yield
        tnx_data = yf.download("^TNX", start=date, progress=False)
        tnx_val = float(tnx_data['Close'].iloc[0].item()) if not tnx_data.empty else 0.0
        
        return {
            "THB_USD": thb_val,
            "US_10Y_YIELD": tnx_val
        }
    except Exception as e:
        logging.error(f"Failed to fetch Macro data: {e}")
        return {"THB_USD": 0.0, "US_10Y_YIELD": 0.0}
