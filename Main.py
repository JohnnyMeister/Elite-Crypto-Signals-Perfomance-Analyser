import json
import re
import csv
import time
import warnings
from datetime import datetime, timedelta

from binance.client import Client
from binance.exceptions import BinanceAPIException
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning, module="_strptime")

# ---------------------------
# Config
# ---------------------------
VERBOSE = True
SLEEP_BETWEEN_BATCHES_SEC = 0.25
MAX_RETRIES = 5
BACKOFF_BASE = 1.5

# Policies and costs
EXIT_POLICY = 'scale_out_equal'  # 'first_tp' | 'last_tp' | 'scale_out_equal'
TAKER_FEE_PCT = 0.0010           # 0.10% typical Spot taker fee
MAKER_FEE_PCT = 0.0010           # adjust if using BNB/vip
USE_TAKER = True                 # assume market execution (worst case)
SLIPPAGE_BPS = 5                 # 5 bps = 0.05% per leg (entry and exit)
INITIAL_EQUITY = 2_000.0         # starting equity for curve
ALLOCATION = 0.15                # fraction of equity per trade (compounded)

API_KEY = "INSERT_API_KEY"
API_SECRET = "INSERT_API_SECRET"
if not API_KEY or not API_SECRET:
    raise ValueError("Please set API_KEY and API_SECRET.")

client = Client(API_KEY, API_SECRET)

def parse_timestamp(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            if ts.endswith('Z'):
                return datetime.fromisoformat(ts[:-1] + '+00:00')
        except Exception:
            return None
    return None

def parse_float_locale(s: str):
    s = s.strip()
    if ',' in s and '.' not in s:
        s2 = s.replace(',', '.')
    else:
        s2 = s.replace(',', '')
    return float(s2)

def is_rate_limit_exception(e: Exception) -> bool:
    msg = str(e)
    code429 = getattr(e, 'status_code', None) == 429
    return code429 or '-1003' in msg or 'Too many requests' in msg

def with_retries(fn, *args, **kwargs):
    delay = 0.75
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except BinanceAPIException as e:
            if is_rate_limit_exception(e) and attempt < MAX_RETRIES:
                time.sleep(delay)
                delay *= BACKOFF_BASE
                continue
            raise

with open('discord_signals.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

if VERBOSE:
    print(f"Processing {len(data)} messages...")

try:
    exchange_info = with_retries(client.get_exchange_info)
    exchange_symbols = {s['symbol'].upper() for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
except BinanceAPIException as e:
    raise ValueError(f"Binance API error: {e}")

# Optional: populate with known failed message IDs if needed
failed_ids_set = set()

SHORT_PATTERNS = [
    re.compile(r'\bShort\b(?!-Term)\b', re.IGNORECASE),
    re.compile(r'\b(Sell[ /]Short|Short[ /]Sell)\b', re.IGNORECASE),
]
SELLING_TARGETS = re.compile(r'\bSelling Targets\b', re.IGNORECASE)
FUTURES_HINT = re.compile(r'\b(perp|perpetual|futures?)\b', re.IGNORECASE)
NUM = r'[\d]+(?:[.,]\d+)?'

def detect_direction(content: str) -> str:
    # Default to long unless explicit short cue is present
    if SELLING_TARGETS.search(content):
        pass
    for pat in SHORT_PATTERNS:
        if pat.search(content):
            return 'short'
    return 'long'

def extract_first_number_after(label_patterns, text):
    for lp in label_patterns:
        m = re.search(lp + r'[\s\-–:]*(' + NUM + r')', text, re.IGNORECASE)
        if m:
            s = m.group(1)
            try:
                v = parse_float_locale(s)
                if v > 0:
                    return v
            except:
                continue
    return None

def parse_entry(content: str):
    labels = [
        r'(?:We\s+Buy|Entry(?:\s+Point)?|Buy\s+around|Entry\s+around)',
        r'(?:Buy\s*@?)',
    ]
    return extract_first_number_after(labels, content)

def parse_stop(content: str):
    labels = [r'(?:Stop\s*Loss|Stop)\b(?:[\s\w]*?)']
    return extract_first_number_after(labels, content)

def parse_targets(content: str, entry: float):
    m = re.search(
        r'(?:Our\s+Selling\s+Targets|Targets|TP)[\s\w:]*?(?:around?:\s*)?([^\n]+?)(?=\n\s*(?:Stop|Stop Loss|Leverage|Expected|$))',
        content,
        re.IGNORECASE
    )
    targets = []
    candidates = []
    if m:
        targets_text = re.sub(r'\s+', ' ', m.group(1)).strip()
        parts = re.split(r'[,\s;/\-–]+', targets_text)
        for p in parts:
            p = p.strip()
            if p:
                candidates.append(p)
    else:
        fallback = re.search(r'(?:Targets|TP)[^\n]*\n?\s*([^\n]+)', content, re.IGNORECASE)
        if fallback:
            targets_text = re.sub(r'\s+', ' ', fallback.group(1)).strip()
            candidates.extend(re.findall(r'[\d.,]+', targets_text))

    for p in candidates:
        s = re.sub(r'[^0-9\.,]', '', p)
        if not s:
            continue
        try:
            t = parse_float_locale(s)
            if t > 0 and (entry is None or abs(t - entry) > 1e-9):
                targets.append(t)
        except:
            pass

    return targets

def infer_direction(entry: float, targets: list, stop: float, initial_dir: str) -> str:
    if not targets or entry is None or stop is None:
        return initial_dir
    above = sum(1 for t in targets if t > entry)
    below = sum(1 for t in targets if t < entry)
    if above + below == 0:
        return initial_dir
    majority_dir = 'long' if above >= below else 'short'
    stop_ok = (stop < entry) if majority_dir == 'long' else (stop > entry)
    if stop_ok:
        return majority_dir
    orig_ok = (stop < entry) if initial_dir == 'long' else (stop > entry)
    return initial_dir if orig_ok else majority_dir

def find_entry_touch_index(highs, lows, entry, direction):
    for i in range(len(highs)):
        hi = float(highs[i]); lo = float(lows[i])
        if lo <= entry <= hi:
            return i
    return None

def evaluate_levels_chronologically(highs, lows, entry, targets, stop, direction, start_index=0):
    n = len(highs)
    targets_sorted = sorted(set(targets), reverse=False) if direction == 'long' else sorted(set(targets), reverse=True)
    hit_idx = {t: None for t in targets_sorted}
    stop_idx = None

    for i in range(start_index, n):
        hi = float(highs[i]); lo = float(lows[i])

        # Stop check
        if direction == 'long':
            if lo <= stop and stop_idx is None:
                stop_idx = i
        else:
            if hi >= stop and stop_idx is None:
                stop_idx = i

        # Targets check
        for t in targets_sorted:
            if hit_idx[t] is not None:
                continue
            if direction == 'long':
                if hi >= t:
                    if stop_idx == i:
                        continue
                    hit_idx[t] = i
            else:
                if lo <= t:
                    if stop_idx == i:
                        continue
                    hit_idx[t] = i

    return hit_idx, stop_idx

def choose_exit(hit_idx, stop_idx, stop, targets, direction, policy):
    # Returns (exit_index, exit_price, exit_reason, partials)
    # partials: list of (weight, price, index) used in 'scale_out_equal'
    targets_sorted = sorted(set(targets), reverse=False) if direction == 'long' else sorted(set(targets), reverse=True)
    valid_hits = [(t, i) for t, i in hit_idx.items() if i is not None]

    if policy in ('first_tp', 'last_tp'):
        if valid_hits:
            if policy == 'first_tp':
                t_sel, i_sel = min(valid_hits, key=lambda x: x[1])
            else:
                if stop_idx is None:
                    t_sel, i_sel = max(valid_hits, key=lambda x: x[1])
                else:
                    valid_pre = [(t, i) for t, i in valid_hits if i < stop_idx]
                    if valid_pre:
                        t_sel, i_sel = max(valid_pre, key=lambda x: x[1])
                    else:
                        return stop_idx, stop, 'stop', []
            return i_sel, t_sel, 'tp', []
        else:
            return stop_idx, stop, 'stop', []
    else:  # scale_out_equal
        partials = []
        if stop_idx is None:
            hits_in_order = sorted(valid_hits, key=lambda x: x[1])
            k = len(hits_in_order)
            if k > 0:
                w = 1.0 / (k + 0)  # all in TPs if no stop
                for t, i in hits_in_order:
                    partials.append((w, t, i))
                return hits_in_order[-1][1], hits_in_order[-1][0], 'scale_tp', partials
            else:
                return None, None, 'no_exit', []
        else:
            hits_pre = sorted([(t, i) for t, i in valid_hits if i < stop_idx], key=lambda x: x[1])
            k = len(hits_pre)
            if k > 0:
                w_tp = 1.0 / (k + 1)  # +1 slice reserved for the stop remainder
                for t, i in hits_pre:
                    partials.append((w_tp, t, i))
                partials.append((w_tp, stop, stop_idx))
                return stop_idx, stop, 'scale_tp_stop', partials
            else:
                return stop_idx, stop, 'stop', [(1.0, stop, stop_idx)]

def fees_total_pct(use_taker=True):
    return (TAKER_FEE_PCT if use_taker else MAKER_FEE_PCT) * 2.0

def slippage_pct():
    return SLIPPAGE_BPS / 10_000.0

def net_return_pct(direction, entry, exit_price):
    fee = fees_total_pct(USE_TAKER)
    slip = slippage_pct()
    if direction == 'long':
        eff_entry = entry * (1 + slip)
        eff_exit  = exit_price * (1 - slip)
        gross = (eff_exit / eff_entry) - 1.0
        return gross - fee
    else:
        eff_entry = entry * (1 - slip)
        eff_exit  = exit_price * (1 + slip)
        gross = (eff_entry / eff_exit) - 1.0
        return gross - fee

def get_klines_1m_paged(symbol, start_dt, end_dt, client, limit=1000):
    interval = Client.KLINE_INTERVAL_1MINUTE
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)
    out = []
    while start_ms < end_ms:
        def call():
            return client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_ms,
                endTime=end_ms,
                limit=limit
            )
        batch = with_retries(call)
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        start_ms = last_open + 60_000
        time.sleep(SLEEP_BETWEEN_BATCHES_SEC)
        if len(batch) < limit:
            break
    return out

signals = []
unparsed = []
trades = []
valid_signals_count = 0
validated_from_failed = 0

for msg in tqdm(data, desc="Parsing signals", unit="msg"):
    msg_id = msg.get('id', 'unknown')
    content = (msg.get('content') or '').strip()
    timestamp_str = msg.get('timestamp', '')

    # Skip "target reached" updates
    if re.match(r'^\s*(Note:\s*)?✅\s*#.*?(reached|hit).*?Target.*$', content, re.IGNORECASE) and len(content) < 150:
        unparsed.append({'id': msg_id, 'reason': 'Target update (not a new signal)', 'content_snippet': content[:120], 'symbol': '', 'notes': 'Intentional skip'})
        continue

    # Skip futures signals
    if FUTURES_HINT.search(content):
        unparsed.append({'id': msg_id, 'reason': 'Futures signal not supported (Spot only)', 'content_snippet': content[:140], 'symbol': '', 'notes': ''})
        continue

    dt = parse_timestamp(timestamp_str)
    if not dt:
        unparsed.append({'id': msg_id, 'reason': 'Invalid/missing timestamp', 'content_snippet': content[:120], 'symbol': '', 'notes': ''})
        continue

    sm = re.search(r'#([A-Z0-9\s]+?)/(USDT|USDC)', content, re.IGNORECASE)
    if not sm:
        unparsed.append({'id': msg_id, 'reason': 'Symbol not found', 'content_snippet': content[:140], 'symbol': '', 'notes': ''})
        continue
    symbol_base = re.sub(r'\s+', '', sm.group(1)).upper()
    quote = sm.group(2).upper()
    symbol = symbol_base + quote

    if symbol not in exchange_symbols:
        unparsed.append({'id': msg_id, 'reason': 'Symbol not listed on Binance', 'content_snippet': '', 'symbol': symbol, 'notes': 'Cannot validate without data'})
        continue

    initial_dir = detect_direction(content)
    entry = parse_entry(content)
    stop = parse_stop(content)
    if entry is None:
        unparsed.append({'id': msg_id, 'reason': 'Entry not found', 'content_snippet': content[:140], 'symbol': symbol, 'notes': ''})
        continue
    if stop is None:
        unparsed.append({'id': msg_id, 'reason': 'Stop not found', 'content_snippet': content[:140], 'symbol': symbol, 'notes': ''})
        continue
    targets = parse_targets(content, entry)
    if not targets:
        unparsed.append({'id': msg_id, 'reason': 'No valid targets', 'content_snippet': content[:200], 'symbol': symbol, 'notes': ''})
        continue

    direction = infer_direction(entry, targets, stop, initial_dir)
    end_date = dt + timedelta(days=30)

    try:
        klines = get_klines_1m_paged(symbol, dt, end_date, client, limit=1000)
    except BinanceAPIException:
        klines = []

    if not klines:
        unparsed.append({'id': msg_id, 'reason': 'No price data (1m)', 'content_snippet': '', 'symbol': symbol, 'notes': f'Window: {dt} to {end_date}'})
        continue

    highs = [float(k[2]) for k in klines]
    lows  = [float(k[3]) for k in klines]

    entry_idx = find_entry_touch_index(highs, lows, entry, direction)
    if entry_idx is None:
        unparsed.append({'id': msg_id, 'reason': 'Entry not triggered within 30 days', 'content_snippet': content[:160], 'symbol': symbol, 'notes': f'entry={entry}'})
        continue

    hit_idx, stop_idx = evaluate_levels_chronologically(highs, lows, entry, targets, stop, direction, start_index=entry_idx)
    exit_index, exit_price, exit_reason, partials = choose_exit(hit_idx, stop_idx, stop, targets, direction, EXIT_POLICY)

    if EXIT_POLICY == 'scale_out_equal' and partials:
        pnl_parts = []
        for w, p, ix in partials:
            pnl_parts.append(w * net_return_pct(direction, entry, p))
        net_pct = sum(pnl_parts)
    else:
        if exit_index is None or exit_price is None:
            # Safety fallback: if nothing was hit, treat as no trade
            unparsed.append({'id': msg_id, 'reason': 'No exit defined', 'content_snippet': content[:160], 'symbol': symbol, 'notes': EXIT_POLICY})
            continue
        net_pct = net_return_pct(direction, entry, exit_price)

    total_targets = len(set(targets))
    num_hit_before_stop = sum(1 for t, i in hit_idx.items() if i is not None and (stop_idx is None or i < stop_idx))
    stop_hit = (exit_reason in ('stop', 'scale_tp_stop'))
    outcome = f"{num_hit_before_stop}/{total_targets} targets hit"
    if stop_hit:
        outcome += " (stop)"

    signals.append({
        'id': msg_id,
        'symbol': symbol,
        'direction': direction,
        'signal_date': dt.isoformat(),
        'entry': round(entry, 6),
        'targets': ','.join(f"{t:.6f}" for t in sorted(set(targets))),
        'stop': round(stop, 6),
        'exit_price': round(exit_price, 6) if exit_price else '',
        'exit_reason': exit_reason,
        'num_targets_hit': int(num_hit_before_stop),
        'total_targets': int(total_targets),
        'stop_hit': stop_hit,
        'net_return_pct': round(net_pct * 100.0, 3),
        'outcome': outcome,
        'policy': EXIT_POLICY
    })

    trades.append(net_pct)

    valid_signals_count += 1
    if msg_id in failed_ids_set:
        validated_from_failed += 1
        tqdm.write(f"[NEW] Valid signal among previous failures #{validated_from_failed}: {symbol} ({direction}) - {outcome} | net={net_pct*100:.2f}%")
    else:
        tqdm.write(f"Valid signal #{valid_signals_count}: {symbol} - {outcome} | net={net_pct*100:.2f}%")

# ---------------------------
# Output CSVs
# ---------------------------
sig_fields = [
    'id','symbol','direction','signal_date','entry','targets','stop',
    'exit_price','exit_reason','num_targets_hit','total_targets','stop_hit',
    'net_return_pct','outcome','policy'
]
if signals:
    with open('signals_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sig_fields)
        writer.writeheader()
        writer.writerows(signals)

unp_fields = ['id','reason','content_snippet','symbol','notes']
with open('unparsed_signals.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=unp_fields)
    writer.writeheader()
    for u in unparsed:
        writer.writerow({k: u.get(k, '') for k in unp_fields})

# ---------------------------
# Aggregate metrics
# ---------------------------
total_valid = len(trades)
wins = [r for r in trades if r > 0]
losses = [r for r in trades if r <= 0]
p_win = (len(wins) / total_valid) if total_valid else 0.0
p_loss = 1.0 - p_win if total_valid else 0.0
avg_win = (sum(wins)/len(wins)) if wins else 0.0
avg_loss = abs(sum(losses)/len(losses)) if losses else 0.0
expectancy = p_win * avg_win - p_loss * avg_loss

# Compounded equity curve with fixed allocation
equity = [INITIAL_EQUITY]
peak = INITIAL_EQUITY
max_dd = 0.0
for r in trades:
    next_eq = equity[-1] * (1.0 + r * ALLOCATION)
    equity.append(next_eq)
    peak = max(peak, next_eq)
    dd = (peak - next_eq) / peak if peak > 0 else 0.0
    max_dd = max(max_dd, dd)

total_return_pct = (equity[-1] / equity[0] - 1.0) * 100.0 if len(equity) > 1 else 0.0

print("\n" + "="*60)
print("FINAL SUMMARY (Net PnL + metrics)")
print("="*60)
print(f"Valid signals: {total_valid}")
print(f"Win rate: {p_win*100:.1f}% | Avg win: {avg_win*100:.2f}% | Avg loss: {avg_loss*100:.2f}%")
print(f"Expectancy: {expectancy*100:.2f}% per trade")
print(f"Total return (compounded, alloc={ALLOCATION:.2f}): {total_return_pct:.2f}%")
print(f"Max Drawdown: {max_dd*100:.2f}%")
print(f"CSV: signals_analysis.csv | unparsed_signals.csv")
print("="*60)
