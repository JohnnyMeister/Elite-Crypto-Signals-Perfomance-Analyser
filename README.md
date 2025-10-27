# 📊 Discord Trading Signals Analyzer

![Python](https://img.shields.io/badge/python-3.x-blue)
![Binance](https://img.shields.io/badge/binance-spot-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚀 Overview

This repository contains a **Python script** that:

* Parses Discord trading signals exported as JSON.
* Validates symbols against **Binance Spot** markets.
* Simulates trade outcomes over a 30‑day window using 1‑minute candles.
* Produces **CSV reports** and aggregated performance metrics.

> Expects a **Discrub export** named `discord_signals.json` in the project root and requires a **Binance API key** to fetch historical klines.

---

## ✨ Features

* **📝 Signal Parsing:** Extracts symbol, direction, entry, stop, and multiple targets using robust regex rules.
* **✅ Spot Validation:** Filters out futures/perpetual signals and inactive symbols.
* **📈 Trade Simulation:** Simulates trade activation/exit on 1-min candles for up to 30 days, accounting for fees, slippage, and configurable exit policies.
* **📂 Output Reports:** Generates CSV files with detailed results and aggregated metrics.

---

## 🛠 Prerequisites

* **Python 3.x** installed.
* **Binance API key & secret** with Spot market read permissions.
* **Discord messages export** in JSON format using **Discrub**.

---

## ⚡ Installation

1. Clone the repository:

```bash
git clone <https://github.com/JohnnyMeister/Elite-Crypto-Signals-Perfomance-Analyser.git>
cd <repo-folder>
```

2. Install dependencies:

```bash
pip install python-binance tqdm
```

---

## 📥 Export JSON with Discrub

1. Install the **Discrub** browser extension.
2. Open Discord in your browser.
3. Launch Discrub via the top-right button or extension icon.
4. Load the desired channel/DM → **Export Loaded Messages → JSON**.
5. Save as `discord_signals.json` in the repository root.

---

## ⚙️ Configuration

Open `analise_sinais.py` and edit:

```python
API_KEY = "your_binance_api_key"
API_SECRET = "your_binance_api_secret"

VERBOSE = True  # Enable progress/info prints
EXIT_POLICY = "first_tp"  # Options: 'first_tp', 'last_tp', 'scale_out_equal'
TAKER_FEE_PCT = 0.04
MAKER_FEE_PCT = 0.02
USE_TAKER = True
```

> Script raises an error if API_KEY or API_SECRET are missing.

---

## ▶️ Usage

Ensure `discord_signals.json` exists, then run:

```bash
python analise_sinais.py
```

Outputs:

* **signals_analysis.csv** → validated signals with performance metrics
* **unparsed_signals.csv** → signals failed parsing
* **Console Summary** → win rate, avg win/loss, expectancy, compounded return, max drawdown

---

## 📄 Input Format

* JSON array of message objects with `id`, `content`, `timestamp`.
* Recognizes patterns like `#SYMBOL/USDT` or `#SYMBOL/USDC`.
* Skips messages that are “target reached” updates or reference futures.

---

## 🗃 Outputs

### 1️⃣ signals_analysis.csv

Fields:

```
id, symbol, direction, signal_date, entry, targets, stop, exit_price,
exit_reason, num_targets_hit, total_targets, stop_hit, net_return_pct,
outcome, policy
```

### 2️⃣ unparsed_signals.csv

Fields:

```
id, reason, content_snippet, symbol, notes
```

### 3️⃣ Console Summary

* Win rate
* Avg win/loss
* Expectancy per trade
* Compounded return
* Max drawdown

---

## ⚙ Configuration Reference

| Parameter     | Description                                 |
| ------------- | ------------------------------------------- |
| VERBOSE       | Show progress/info prints                   |
| EXIT_POLICY   | 'first_tp', 'last_tp', or 'scale_out_equal' |
| TAKER_FEE_PCT | Fee percentage for taker orders             |
| MAKER_FEE_PCT | Fee percentage for maker orders             |
| USE_TAKER     | Market (taker) or maker execution           |

---

## 🔗 Notes

* Full pipeline: **Discord signal export → validation → trade simulation → performance reporting**
* Designed for **Binance Spot** only




