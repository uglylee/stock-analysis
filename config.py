import os

# Proxy configuration
USE_PROXY = False  # Set to True to enable proxy (e.g., for 7890)
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7890
PROXY_TYPE = "http"  # "http" or "socks5"
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
SOCKS5_URL = f"socks5://{PROXY_HOST}:{PROXY_PORT}"

if USE_PROXY:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# Data settings
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOOKBACK_YEARS = 2
SURGE_THRESHOLD = 1.0  # 100% gain
SURGE_WINDOW_DAYS = 20  # ~1 month trading days
PRE_SURGE_WINDOW = 60  # days before surge for feature extraction

# Cache files
STOCK_LIST_CACHE = os.path.join(DATA_DIR, "stock_list.csv")
DAILY_DATA_DIR = os.path.join(DATA_DIR, "daily")
ANALYSIS_RESULT = os.path.join(DATA_DIR, "analysis_result.csv")
FEATURES_CACHE = os.path.join(DATA_DIR, "features.csv")
PREDICTION_RESULT = os.path.join(DATA_DIR, "predictions.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DAILY_DATA_DIR, exist_ok=True)
