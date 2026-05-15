"""
One-click runner for the A-share surge stock analysis system.

Steps:
  1. python run.py collect   — Download all stock data (takes a while)
  2. python run.py analyze   — Find surge stocks and extract features
  3. python run.py predict   — Screen current market for matching stocks
  4. python run.py web       — Start the web dashboard
  5. python run.py all       — Run steps 1-4 sequentially
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from data_collector import ensure_proxy


def cmd_collect():
    from data_collector import get_stock_list, download_all_stocks
    ensure_proxy()
    stocks = get_stock_list()
    if stocks.empty:
        print("[!] Failed to get stock list. Check network connection.")
        return
    print(f"[*] Got {len(stocks)} stocks, starting download...")
    data = download_all_stocks(stocks, max_workers=8)
    print(f"[+] Collected data for {len(data)} stocks")


def cmd_analyze():
    from data_collector import get_stock_list, load_cached_data
    from analyzer import analyze_all_stocks, compute_common_features

    stock_info = get_stock_list()
    stock_data = load_cached_data()
    if not stock_data:
        print("[!] No cached data found. Run 'collect' first.")
        return

    result = analyze_all_stocks(stock_data, stock_info)
    if not result.empty:
        compute_common_features(result)
        print(f"\n[+] Found {len(result)} surge events")
        print(f"[+] Average gain: {result['gain_pct'].mean():.1f}%")
        top = result.nlargest(10, "gain_pct")
        for _, row in top.iterrows():
            print(f"    {row['code']} {row['name']}: {row['gain_pct']:.0f}%")


def cmd_predict():
    from predictor import run_prediction
    pred = run_prediction()
    if not pred.empty:
        print(f"\n[+] Top predictions saved to data/predictions.csv")


def cmd_web():
    from app import app
    print("\n[*] Starting web server at http://127.0.0.1:5000")
    print("[*] Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=True)


def cmd_all():
    cmd_collect()
    print("\n" + "=" * 60 + "\n")
    cmd_analyze()
    print("\n" + "=" * 60 + "\n")
    cmd_predict()
    print("\n" + "=" * 60 + "\n")
    print("[*] All analysis complete! Starting web server...\n")
    cmd_web()


USAGE = """
Usage:
  python run.py collect    — Download A-share daily data
  python run.py analyze    — Find stocks with >100% monthly gains
  python run.py predict    — Screen current market for matching stocks
  python run.py web        — Start the web dashboard
  python run.py all        — Run all steps and start web server
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1].lower()
    commands = {
        "collect": cmd_collect,
        "analyze": cmd_analyze,
        "predict": cmd_predict,
        "web": cmd_web,
        "all": cmd_all,
    }

    if cmd in commands:
        commands[cmd]()
    else:
        print(f"Unknown command: {cmd}")
        print(USAGE)
