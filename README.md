# Elite Crypto Signals Performance Analyser 📊



## 🧩 Using Discrub to Obtain JSON Data

The project requires a **JSON export** of your crypto signal messages from Discord. You can easily obtain this file using the **Discrub** browser extension.

### 🧠 What is Discrub?

[**Discrub**](https://github.com/superseriousbusiness/discrub) is a browser extension that allows you to **export Discord channel messages** into a clean JSON file — perfect for analysis.

### 📥 How to Use Discrub

1. **Install the Discrub extension**:

   * Available for **Chrome**, **Edge**, and **Firefox** (search for “Discrub” in your browser’s extension store).
2. **Open the Discord channel** that contains your crypto signals.
3. **Click the Discrub icon** in your browser toolbar.
4. Choose **Export as JSON**.
5. Wait for the export to complete — this may take a few minutes depending on the number of messages.
6. Save the resulting file as `discord_signals.json` in your project’s root directory.

That’s it! You now have a ready-to-use JSON file that can be processed by the Elite Crypto Signals Performance Analyser.

---

## 🛠️ Prerequisites

* Python 3.x installed on your system.
* A valid API Key & Secret for Binance (Spot read permissions).
* Exported signals JSON file (`discord_signals.json`) obtained using **Discrub**.

---

## ⚡ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/JohnnyMeister/Elite-Crypto-Signals-Perfomance-Analyser.git
   cd Elite-Crypto-Signals-Perfomance-Analyser
   ```
2. Install dependencies:

   ```bash
   pip install python-binance tqdm
   ```

---

## 🔧 Configuration

Open the main script (for example `Main.py`) and adjust settings such as:

```python
API_KEY = "your_binance_api_key"
API_SECRET = "your_binance_api_secret"

VERBOSE = True              # Show detailed progress/info
EXIT_POLICY = "first_tp"    # Options: 'first_tp', 'last_tp', 'scale_out_equal'
TAKER_FEE_PCT = 0.04        # Taker fee percentage
MAKER_FEE_PCT = 0.02        # Maker fee percentage
USE_TAKER = True            # Use taker execution if True
```

Be sure your API credentials are correctly set; the script will error out otherwise.

---

## ▶️ Usage

With the `discord_signals.json` file in place and your API credentials configured, run:

```bash
python Main.py
```

**Outputs produced:**

* `signals_analysis.csv` → validated signals with performance metrics
* `unparsed_signals.csv` → signals that failed parsing
* Console summary with statistics: win rate, average win/loss, expectancy per trade, compounded return, max drawdown

---

## 📅 Input Format

* A JSON array of exported message objects (with fields like `id`, `content`, `timestamp`).
* Supports patterns such as `#SYMBOL/USDT` or `#SYMBOL/USDC`.
* Skips messages that are simple “target reached” updates or refer to futures/perpetuals.

---

## 📂 Outputs Explained

### 1️⃣ `signals_analysis.csv`

Fields include: `id`, `symbol`, `direction`, `signal_date`, `entry`, `targets`, `stop`, `exit_price`, `exit_reason`, `num_targets_hit`, `total_targets`, `stop_hit`, `net_return_pct`, `outcome`, `policy`.

### 2️⃣ `unparsed_signals.csv`

Fields: `id`, `reason` (why parsing failed), `content_snippet`, `symbol`, `notes`.

### 3️⃣ Console Summary

Reports:

* Win rate
* Average win / average loss
* Expectancy per trade
* Compounded return
* Maximum drawdown

---

## 🧭 Notes

* The full pipeline: Export signals from Discord → validate them → simulate trades → generate performance reports.
* Designed for Binance **Spot** market only (not futures/perpetual).
* Be sure to review your exit policy and fee/slippage assumptions — they will strongly affect results.

---

