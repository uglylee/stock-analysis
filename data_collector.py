"""
A-share stock data collector with proxy support (port 7890).
Downloads and caches daily price data for all A-share stocks.
Uses Tencent kline API and programmatic stock code generation.
"""
import os
import re
import time
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    USE_PROXY, PROXY_URL, DATA_DIR, DAILY_DATA_DIR, STOCK_LIST_CACHE, LOOKBACK_YEARS
)


def _create_session() -> requests.Session:
    session = requests.Session()
    if USE_PROXY:
        session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://gu.qq.com/",
    })
    retry = Retry(total=2, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def ensure_proxy():
    if not USE_PROXY:
        return
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ[key] = PROXY_URL


_global_session = None

def get_session() -> requests.Session:
    global _global_session
    if _global_session is None:
        _global_session = _create_session()
    return _global_session


# ---- Stock Code Generation ----

def generate_stock_codes() -> pd.DataFrame:
    """
    Generate A-share stock codes programmatically.
    Shanghai: 600xxx-605xxx, 688xxx-689xxx (STAR)
    Shenzhen: 000xxx-004xxx, 300xxx-301xxx (ChiNext), 002xxx-003xxx
    Beijing: 8xxxxx
    Returns DataFrame with code + placeholder name.
    """
    codes = set()

    # Shanghai main board: 600000 - 605999
    for i in range(600000, 606000):
        codes.add(str(i))
    # Shanghai STAR: 688000 - 689999
    for i in range(688000, 690000):
        codes.add(str(i))
    # Shenzhen main: 000001 - 004999
    for i in range(1, 5000):
        codes.add(str(i).zfill(6))
    # Shenzhen SME: 002000 - 003999
    for i in range(2000, 4000):
        codes.add(str(i).zfill(6))
    # Shenzhen ChiNext: 300000 - 301999
    for i in range(300000, 302000):
        codes.add(str(i))

    df = pd.DataFrame({"code": sorted(codes), "name": ""})
    df.to_csv(STOCK_LIST_CACHE, index=False)
    print(f"[*] Generated {len(df)} potential stock codes")
    return df


def get_stock_list(force_refresh: bool = False) -> pd.DataFrame:
    """Get all A-share stock codes via Sina API, cached locally."""
    if not force_refresh and os.path.exists(STOCK_LIST_CACHE):
        df = pd.read_csv(STOCK_LIST_CACHE, dtype={"code": str})
        if len(df) > 1000:
            print(f"[*] Loaded {len(df)} cached stock codes")
            return df

    ensure_proxy()
    print("[*] Fetching A-share stock list via Sina API...")
    session = get_session()
    all_stocks = []

    for page in range(1, 200):
        url = (
            f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
            f"/Market_Center.getHQNodeData"
            f"?page={page}&num=40&sort=symbol&asc=1&node=hs_a"
        )
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                break
            text = r.text.strip()
            if not text or len(text) < 10 or text.startswith("null"):
                break
            data = json.loads(text)
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            for item in data:
                all_stocks.append({
                    "code": str(item.get("code", "")).zfill(6),
                    "name": item.get("name", ""),
                })
            if len(data) < 40:
                break  # Last page
            if page % 20 == 0:
                print(f"    Page {page}: {len(all_stocks)} stocks so far")
        except Exception as e:
            print(f"[!] Error on page {page}: {e}")
            break

    if len(all_stocks) < 500:
        print(f"[!] Only got {len(all_stocks)} stocks, using generated codes fallback")
        return generate_stock_codes()

    df = pd.DataFrame(all_stocks).drop_duplicates(subset=["code"])
    df.to_csv(STOCK_LIST_CACHE, index=False)
    print(f"[+] Got {len(df)} stocks via Sina, cached to {STOCK_LIST_CACHE}")
    return df


# ---- Daily K-line via Tencent API ----

def _tencent_kline_url(code: str) -> str:
    """Build Tencent K-line API URL."""
    if code.startswith(("6", "9")):
        market = "sh"
    elif code.startswith(("0", "3", "2")):
        market = "sz"
    elif code.startswith("8"):
        market = "bj"
    else:
        market = "sz"
    return (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={market}{code},day,,,{LOOKBACK_YEARS * 280},qfq"
    )


def _tencent_batch_quote_url(codes: list[str]) -> str:
    """Build Tencent batch quote URL for verifying stock existence."""
    code_strs = []
    for code in codes:
        if code.startswith(("6", "9")):
            code_strs.append(f"sh{code}")
        else:
            code_strs.append(f"sz{code}")
    return f"https://qt.gtimg.cn/q={','.join(code_strs)}"


def download_one_stock(code: str) -> pd.DataFrame | None:
    """Download daily data for a single stock via Tencent API."""
    cache_file = os.path.join(DAILY_DATA_DIR, f"{code}.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty and len(df) > 20:
                return df
        except Exception:
            pass

    session = get_session()

    for attempt in range(3):
        try:
            url = _tencent_kline_url(code)
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                if attempt < 2:
                    time.sleep(1 + attempt)
                continue

            data = r.json()

            # Determine market prefix
            if code.startswith(("6", "9")):
                market = "sh"
            elif code.startswith("8"):
                market = "bj"
            else:
                market = "sz"

            key = f"{market}{code}"
            stock_data = data.get("data", {}).get(key)

            if not stock_data:
                return None

            kline_data = stock_data.get("qfqday") or stock_data.get("day")
            if not kline_data or len(kline_data) < 20:
                return None

            rows = []
            for item in kline_data:
                if len(item) >= 6:
                    try:
                        rows.append({
                            "date": item[0],
                            "open": float(item[1]),
                            "close": float(item[2]),
                            "high": float(item[3]),
                            "low": float(item[4]),
                            "volume": float(item[5]),
                            "amount": float(item[6]) if len(item) > 6 and not isinstance(item[6], dict) else 0.0,
                        })
                    except (ValueError, TypeError):
                        continue

            df = pd.DataFrame(rows)
            if df.empty:
                return None

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

            # Derived columns
            df["pct_change"] = df["close"].pct_change() * 100
            df["change"] = df["close"].diff()
            df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100
            df["turnover"] = 0.0

            # Save cache
            df.to_csv(cache_file)
            return df

        except Exception:
            if attempt < 2:
                time.sleep(1 + attempt)

    return None


def download_all_stocks(
    stock_list: pd.DataFrame, max_workers: int = 10
) -> dict[str, pd.DataFrame]:
    """Download daily data for all stocks concurrently."""
    codes = stock_list["code"].tolist()
    results = {}
    total = len(codes)
    done = 0
    failed = 0

    print(f"[*] Downloading daily data for {total} stocks ({max_workers} threads)...")
    ensure_proxy()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one_stock, code): code
                   for code in codes}

        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[code] = df
                else:
                    failed += 1
            except Exception:
                failed += 1
            if done % 1000 == 0:
                print(f"    Progress: {done}/{total} | "
                      f"cached: {len(results)} | failed: {failed}")

    print(f"[+] Downloaded data for {len(results)}/{total} stocks "
          f"(failed: {failed})")
    return results


def load_cached_data() -> dict[str, pd.DataFrame]:
    """Load all cached daily data from disk."""
    results = {}
    if not os.path.exists(DAILY_DATA_DIR):
        return results
    files = [f for f in os.listdir(DAILY_DATA_DIR) if f.endswith(".csv")]
    for f in files:
        code = f.replace(".csv", "")
        try:
            df = pd.read_csv(
                os.path.join(DAILY_DATA_DIR, f), index_col=0, parse_dates=True
            )
            if not df.empty:
                results[code] = df
        except Exception:
            pass
    print(f"[*] Loaded {len(results)} cached stock data files")
    return results


if __name__ == "__main__":
    ensure_proxy()

    # Test connection
    session = get_session()
    r = session.get(
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        "?param=sz000001,day,,,500,qfq",
        timeout=15
    )
    print(f"Connection test: status={r.status_code}")

    # Get stock list
    stocks = get_stock_list(force_refresh=True)
    print(f"Stock count: {len(stocks)}")

    # Test single stock download
    print("\n[*] Testing single stock download...")
    df = download_one_stock("000001")
    if df is not None:
        print(f"  000001: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Last close: {df['close'].iloc[-1]}")
    else:
        print("  000001: FAILED")
