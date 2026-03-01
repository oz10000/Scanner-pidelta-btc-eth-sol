import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from itertools import product
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN
# ============================================================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h"]
LOOKBACK_DAYS = 60                # Datos para backtest (suficiente para 60 días)
SCAN_LOOKBACK = 20                 # Escáner mira últimos 20 días (cubre señales recientes)
MC_ITER = 200
DATA_CACHE = {}
ATR_CACHE = {}
PIDELTA_CACHE = {}

# Parámetros del escáner
TENSION_QUANTILE_SCAN = 0.85
SCAN_K_VALUES = [3, 5, 8, 13]
MIN_FUTURE_VELAS = 20              # Mínimo de velas hacia adelante

# Grid de optimización
PARAM_GRID = {
    'tp_atr': [1, 2, 3, 5, 8],
    'sl_atr': [1, 2, 3, 5],
    'atr_window': [7, 14, 21],
    'tension_quantile': [0.5, 0.6, 0.7, 0.8, 0.9],
    'pidelta_window': [5, 8, 13, 21]
}
PARAM_NAMES = list(PARAM_GRID.keys())

MAX_WORKERS = 8
MAX_SIGNALS_FOR_OPT = 200

# ============================================================
# DESCARGA DE DATOS (con reintento y paginación implícita)
# ============================================================
def fetch_klines(symbol, interval, days=LOOKBACK_DAYS):
    key = f"{symbol}_{interval}"
    if key in DATA_CACHE:
        return DATA_CACHE[key]

    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start, "endTime": end, "limit": 1500}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","num_trades","taker_base_vol","taker_quote_vol","ignore"
        ])
        df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df = df[["open","high","low","close","volume"]]
        DATA_CACHE[key] = df
        return df
    except Exception as e:
        print(f"Error descargando {symbol} {interval}: {e}")
        return None

def get_atr(symbol, tf, window):
    key = (symbol, tf, window)
    if key in ATR_CACHE:
        return ATR_CACHE[key]
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < window:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(window).mean()
    ATR_CACHE[key] = atr_series
    return atr_series

def get_pidelta(symbol, tf, window):
    key = (symbol, tf, window)
    if key in PIDELTA_CACHE:
        return PIDELTA_CACHE[key]
    df = fetch_klines(symbol, tf)
    if df is None or len(df) < window:
        return None
    price = df["close"]
    rets = price.pct_change().fillna(0)
    P_struct = rets.rolling(window).mean()
    P_hist = rets.ewm(span=window).mean()
    pidelta = P_struct - P_hist
    PIDELTA_CACHE[key] = pidelta
    return pidelta

# ============================================================
# INDICADORES Y ESCÁNER MEJORADO
# ============================================================
def tension_235(series):
    ema2 = series.ewm(span=2).mean()
    ema3 = series.ewm(span=3).mean()
    ema5 = series.ewm(span=5).mean()
    return (ema2 - ema3).abs() + (ema3 - ema5).abs()

def estimate_winrate(price, tension, dt, direction, k, window=50):
    """Estima winrate histórico usando puntos de alta tensión anteriores."""
    threshold = tension.quantile(0.85)
    similar = tension[tension >= threshold].index
    if len(similar) < 5:
        return 0.5
    similar = [t for t in similar if t < dt][-50:]  # últimos 50 anteriores
    wins = 0
    total = 0
    for t in similar:
        try:
            idx = price.index.get_loc(t)
            if idx + k < len(price):
                ret = price.iloc[idx+k] - price.iloc[idx]
                if direction == 'LONG' and ret > 0:
                    wins += 1
                elif direction == 'SHORT' and ret < 0:
                    wins += 1
                total += 1
        except:
            continue
    return wins / total if total > 0 else 0.5

def scan_symbol_tf(symbol, tf):
    df = fetch_klines(symbol, tf, days=SCAN_LOOKBACK)
    if df is None or len(df) < 100:
        return []
    price = df['close']
    tension = tension_235(price)
    threshold = tension.quantile(TENSION_QUANTILE_SCAN)
    high_tension_points = tension[tension >= threshold].index

    signals = []
    for dt in high_tension_points:
        try:
            idx = price.index.get_loc(dt)
        except:
            continue
        if idx + MIN_FUTURE_VELAS >= len(price):
            continue
        c2 = price.ewm(span=2).mean().iloc[idx]
        c3 = price.ewm(span=3).mean().iloc[idx]
        c5 = price.ewm(span=5).mean().iloc[idx]
        for k in SCAN_K_VALUES:
            if idx + k >= len(price):
                continue
            future_ret = price.iloc[idx + k] - price.iloc[idx]
            ret_pct = future_ret / price.iloc[idx]
            if future_ret > 0:  # LONG
                winrate_est = estimate_winrate(price, tension, dt, 'LONG', k)
                signals.append({
                    'Symbol': symbol,
                    'TF': tf,
                    'open_time': dt,
                    'Direction': 'LONG',
                    'tension': tension.loc[dt],
                    'edge': ret_pct,
                    'winrate': winrate_est,
                    'k': k,
                    'score': ret_pct * winrate_est,
                    'price_entry': price.iloc[idx],
                    'c2': c2, 'c3': c3, 'c5': c5
                })
            if future_ret < 0:  # SHORT
                winrate_est = estimate_winrate(price, tension, dt, 'SHORT', k)
                signals.append({
                    'Symbol': symbol,
                    'TF': tf,
                    'open_time': dt,
                    'Direction': 'SHORT',
                    'tension': tension.loc[dt],
                    'edge': -ret_pct,
                    'winrate': winrate_est,
                    'k': k,
                    'score': (-ret_pct) * winrate_est,
                    'price_entry': price.iloc[idx],
                    'c2': c2, 'c3': c3, 'c5': c5
                })
    return signals

def run_scanner():
    all_signals = []
    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            print(f"🔍 Escaneando {sym} {tf}...")
            sigs = scan_symbol_tf(sym, tf)
            all_signals.extend(sigs)
    df_signals = pd.DataFrame(all_signals)
    if len(df_signals) > 0:
        df_signals = df_signals.sort_values('score', ascending=False)
        df_signals.to_csv("escaneo_filtrado.csv", index=False)
        print(f"✅ Escáner completado. {len(df_signals)} señales guardadas.")
    else:
        print("⚠️ No se generaron señales.")
    return df_signals

# ============================================================
# BACKTEST DE UNA SEÑAL INDIVIDUAL (con verificación de rango)
# ============================================================
def backtest_signal(symbol, tf, signal_time, direction, tp_atr, sl_atr, atr_window, max_lookahead=30):
    df = fetch_klines(symbol, tf)
    if df is None:
        return {'error': 'No data'}

    # Verificar que la fecha esté dentro del rango del DataFrame
    if signal_time < df.index[0] or signal_time > df.index[-1]:
        return {'error': f'Signal time {signal_time} out of range [{df.index[0]}, {df.index[-1]}]'}

    try:
        idx = df.index.get_loc(signal_time, method='nearest')
    except:
        return {'error': 'Time not found even with nearest'}

    atr_series = get_atr(symbol, tf, atr_window)
    if atr_series is None:
        return {'error': 'No ATR'}
    if idx < atr_window or idx >= len(df) - 1:
        return {'error': f'Index {idx} out of valid range (need >= {atr_window} and < {len(df)-1})'}

    entry_price = df['close'].iloc[idx]
    atr_val = atr_series.iloc[idx]
    if pd.isna(atr_val) or atr_val == 0:
        return {'error': 'ATR is NaN/zero'}

    if direction == 'LONG':
        target = entry_price + tp_atr * atr_val
        stop = entry_price - sl_atr * atr_val
    else:
        target = entry_price - tp_atr * atr_val
        stop = entry_price + sl_atr * atr_val

    for i in range(1, min(max_lookahead, len(df)-idx-1)):
        high = df['high'].iloc[idx+i]
        low = df['low'].iloc[idx+i]
        if direction == 'LONG':
            if high >= target:
                ret = (target - entry_price) / entry_price
                return {
                    'exit_type': 'TP', 'bars': i, 'return': ret, 'exit_price': target,
                    'entry_price': entry_price, 'atr': atr_val,
                    'target_price': target, 'stop_price': stop, 'error': None
                }
            if low <= stop:
                ret = (stop - entry_price) / entry_price
                return {
                    'exit_type': 'SL', 'bars': i, 'return': ret, 'exit_price': stop,
                    'entry_price': entry_price, 'atr': atr_val,
                    'target_price': target, 'stop_price': stop, 'error': None
                }
        else:
            if low <= target:
                ret = (entry_price - target) / entry_price
                return {
                    'exit_type': 'TP', 'bars': i, 'return': ret, 'exit_price': target,
                    'entry_price': entry_price, 'atr': atr_val,
                    'target_price': target, 'stop_price': stop, 'error': None
                }
            if high >= stop:
                ret = (entry_price - stop) / entry_price
                return {
                    'exit_type': 'SL', 'bars': i, 'return': ret, 'exit_price': stop,
                    'entry_price': entry_price, 'atr': atr_val,
                    'target_price': target, 'stop_price': stop, 'error': None
                }
    final_idx = min(idx+max_lookahead, len(df)-1)
    final_price = df['close'].iloc[final_idx]
    ret = (final_price - entry_price) / entry_price if direction=='LONG' else (entry_price - final_price) / entry_price
    return {
        'exit_type': 'NONE', 'bars': final_idx - idx, 'return': ret, 'exit_price': final_price,
        'entry_price': entry_price, 'atr': atr_val,
        'target_price': target, 'stop_price': stop, 'error': None
    }

# ============================================================
# MÉTRICAS DE PORTAFOLIO
# ============================================================
def portfolio_metrics(trades, daily_corr):
    if len(trades) < 3:
        return None, None, None, None, -np.inf
    df_trades = pd.DataFrame(trades)
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_trades['date'] = df_trades['exit_time'].dt.date
    daily_returns = df_trades.groupby('date')['return'].sum().sort_index()
    if len(daily_returns) == 0:
        return None, None, None, None, -np.inf
    total_return = daily_returns.sum()
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-9)
    equity = daily_returns.cumsum()
    peak = equity.expanding().max()
    dd = (equity - peak).min()
    max_dd = abs(dd)
    returns_array = df_trades['return'].values
    n = len(returns_array)
    perm_returns = []
    for _ in range(MC_ITER):
        sign_perm = np.random.choice([-1, 1], size=n)
        perm_total = (returns_array * sign_perm).sum()
        perm_returns.append(perm_total)
    perm_returns = np.array(perm_returns)
    z_score = (total_return - perm_returns.mean()) / (perm_returns.std() + 1e-9)
    score = total_return * sharpe * (1 - max_dd) * (1 - daily_corr) * (1 + max(0, z_score)/3)
    return total_return, sharpe, max_dd, z_score, score

def get_daily_correlation(symbols):
    daily_data = {}
    for sym in symbols:
        df = fetch_klines(sym, '1d', days=LOOKBACK_DAYS)
        if df is not None:
            daily_data[sym] = df['close'].pct_change().dropna()
    common_dates = None
    for sym in symbols:
        if sym in daily_data:
            if common_dates is None:
                common_dates = set(daily_data[sym].index)
            else:
                common_dates = common_dates.intersection(daily_data[sym].index)
    if not common_dates:
        return 0.5
    common_dates = sorted(common_dates)
    rets_df = pd.DataFrame({sym: daily_data[sym].loc[common_dates] for sym in symbols if sym in daily_data})
    corr_matrix = rets_df.corr()
    off_diag = []
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            if symbols[i] in corr_matrix and symbols[j] in corr_matrix:
                off_diag.append(abs(corr_matrix.loc[symbols[i], symbols[j]]))
    return np.mean(off_diag) if off_diag else 0.5

# ============================================================
# OPTIMIZACIÓN PARALELA
# ============================================================
def evaluate_combo(combo, signals_df, time_col, daily_corr):
    params = dict(zip(PARAM_NAMES, combo))
    tp_atr, sl_atr, atr_win, tens_q, pid_win = params.values()

    threshold = signals_df['tension'].quantile(tens_q)
    filtered = signals_df[signals_df['tension'] >= threshold]
    if filtered.empty:
        return None

    trades = []
    for _, sig in filtered.iterrows():
        df = fetch_klines(sig['Symbol'], sig['TF'])
        if df is None:
            continue
        # Verificar que la fecha esté en el rango
        if sig[time_col] < df.index[0] or sig[time_col] > df.index[-1]:
            continue
        try:
            idx_s = df.index.get_loc(sig[time_col], method='nearest')
        except:
            continue
        if idx_s + 30 >= len(df):
            continue

        res = backtest_signal(
            symbol=sig['Symbol'],
            tf=sig['TF'],
            signal_time=sig[time_col],
            direction=sig['Direction'],
            tp_atr=tp_atr,
            sl_atr=sl_atr,
            atr_window=atr_win
        )
        if res and res['error'] is None:
            trades.append({
                'return': res['return'],
                'exit_time': df.index[min(idx_s+res['bars'], len(df)-1)],
                'symbol': sig['Symbol'],
                'direction': sig['Direction'],
                'exit_type': res['exit_type']
            })

    if len(trades) < 3:
        return None

    total_ret, sharpe, mdd, z, score = portfolio_metrics(trades, daily_corr)
    if score == -np.inf:
        return None

    return {**params, 'num_trades': len(trades), 'total_return': total_ret,
            'sharpe': sharpe, 'max_drawdown': mdd, 'z_score': z, 'score': score}

def optimize_parameters(signals_df):
    if signals_df.empty:
        return None

    time_col = None
    for col in ['open_time', 'timestamp', 'datetime', 'time']:
        if col in signals_df.columns:
            time_col = col
            break
    if time_col is None:
        time_col = signals_df.columns[0]
    signals_df[time_col] = pd.to_datetime(signals_df[time_col])
    if 'tension' not in signals_df.columns:
        signals_df['tension'] = 1.0

    if len(signals_df) > MAX_SIGNALS_FOR_OPT:
        signals_df = signals_df.head(MAX_SIGNALS_FOR_OPT).copy()
        print(f"📉 Usando solo las top {MAX_SIGNALS_FOR_OPT} señales para optimización.")

    daily_corr = get_daily_correlation(SYMBOLS)
    print(f"📊 Correlación media diaria entre activos: {daily_corr:.3f}")

    combinations = list(product(*PARAM_GRID.values()))
    total = len(combinations)
    print(f"🔍 Evaluando {total} combinaciones en paralelo ({MAX_WORKERS} workers)...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_combo = {executor.submit(evaluate_combo, combo, signals_df, time_col, daily_corr): combo
                           for combo in combinations}
        for i, future in enumerate(as_completed(future_to_combo)):
            combo = future_to_combo[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
                    print(f"   ✅ {combo} -> score={res['score']:.4f} (trades={res['num_trades']})")
                if (i+1) % 100 == 0:
                    print(f"   Progreso: {i+1}/{total} combinaciones evaluadas.")
            except Exception as e:
                print(f"   ❌ Error en {combo}: {e}")

    if not results:
        print("❌ No se encontraron combinaciones válidas.")
        return None

    best = max(results, key=lambda x: x['score'])
    print("\n🏆 Mejor combinación:")
    for k, v in best.items():
        print(f"   {k}: {v}")
    df_opt = pd.DataFrame(results)
    df_opt.to_csv("optimization_results.csv", index=False)
    return best

# ============================================================
# ANÁLISIS DETALLADO DE TOP SEÑALES
# ============================================================
def analyze_top_signals(signals_df, best_params):
    if signals_df.empty:
        print("No hay señales para analizar.")
        return
    long_signals = signals_df[signals_df['Direction'] == 'LONG'].sort_values('score', ascending=False).head(3)
    short_signals = signals_df[signals_df['Direction'] == 'SHORT'].sort_values('score', ascending=False).head(3)
    top_signals = pd.concat([long_signals, short_signals])

    print("\n" + "="*80)
    print("📊 TOP 3 SEÑALES LONG Y TOP 3 SHORT - ANÁLISIS COMPLETO")
    print("="*80)

    for _, sig in top_signals.iterrows():
        print(f"\n🔹 {sig['Symbol']} {sig['TF']} {sig['Direction']} | Fecha: {sig['open_time']}")
        print(f"   Score: {sig['score']:.6f} | Tensión: {sig['tension']:.4f} | Edge: {sig['edge']:.4%} | Winrate estimado: {sig['winrate']:.2%}")
        print(f"   Precio entrada: {sig['price_entry']:.2f} | C2: {sig['c2']:.2f} | C3: {sig['c3']:.2f} | C5: {sig['c5']:.2f}")

        # PiDelta
        pid_win = best_params['pidelta_window']
        pidelta_series = get_pidelta(sig['Symbol'], sig['TF'], pid_win)
        if pidelta_series is not None:
            try:
                pidx = pidelta_series.index.get_loc(sig['open_time'], method='nearest')
                pid_val = pidelta_series.iloc[pidx]
                print(f"   PiDelta (ventana {pid_win}): {pid_val:.6f}")
            except:
                print("   PiDelta: No disponible en fecha exacta")
        else:
            print("   PiDelta: No disponible")

        # Backtest individual
        res = backtest_signal(
            symbol=sig['Symbol'],
            tf=sig['TF'],
            signal_time=pd.to_datetime(sig['open_time']),
            direction=sig['Direction'],
            tp_atr=best_params['tp_atr'],
            sl_atr=best_params['sl_atr'],
            atr_window=best_params['atr_window']
        )
        if res and res['error'] is None:
            print(f"   ✅ Backtest: {res['exit_type']} en {res['bars']} velas | Retorno: {res['return']:.4%}")
            print(f"      ATR: {res['atr']:.2f} | Target: {res['target_price']:.2f} | Stop: {res['stop_price']:.2f}")
            amp_tp = abs(res['target_price'] - res['entry_price']) / res['entry_price']
            amp_sl = abs(res['stop_price'] - res['entry_price']) / res['entry_price']
            print(f"      Amplitud TP: {amp_tp:.2%} | SL: {amp_sl:.2%}")
        else:
            motivo = res['error'] if res else "Desconocido"
            print(f"   ❌ Backtest falló: {motivo}")

    # Análisis de correlación BTC/ETH
    best_global = signals_df.iloc[0]
    tf_princ = best_global['TF']
    print("\n" + "="*80)
    print("🔗 ANÁLISIS DE CONCURRENCIA BTC/ETH (en TF de la mejor señal)")
    print("="*80)
    pid_win = best_params['pidelta_window']
    pidelta_btc = get_pidelta('BTCUSDT', tf_princ, pid_win)
    pidelta_eth = get_pidelta('ETHUSDT', tf_princ, pid_win)
    if pidelta_btc is not None and pidelta_eth is not None:
        common_idx = pidelta_btc.index.intersection(pidelta_eth.index)
        pidelta_btc = pidelta_btc.loc[common_idx]
        pidelta_eth = pidelta_eth.loc[common_idx]
        corr_pidelta = pidelta_btc.corr(pidelta_eth)
        print(f"Correlación de PiDelta (ventana {pid_win}): {corr_pidelta:.3f}")
    else:
        print("No se pudo calcular PiDelta para ambos.")

    df_btc = fetch_klines('BTCUSDT', tf_princ)
    df_eth = fetch_klines('ETHUSDT', tf_princ)
    if df_btc is not None and df_eth is not None:
        ret_btc = df_btc['close'].pct_change().dropna()
        ret_eth = df_eth['close'].pct_change().dropna()
        common_ret = ret_btc.index.intersection(ret_eth.index)
        corr_ret = ret_btc.loc[common_ret].corr(ret_eth.loc[common_ret])
        print(f"Correlación de retornos: {corr_ret:.3f}")
        print(f"Concurrencia {'ALTA' if corr_ret > 0.5 else 'BAJA'}")

    print("\n📋 INFORME DE CONFIANZA")
    print(f"Parámetros óptimos: TP={best_params['tp_atr']}*ATR, SL={best_params['sl_atr']}*ATR, ATR window={best_params['atr_window']}")
    print("="*80)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🚀 SISTEMA INTEGRADO: ESCÁNER + OPTIMIZACIÓN + TOP SEÑALES")
    print("\n--- ESCÁNER ---")
    signals = run_scanner()
    if signals.empty:
        print("No se generaron señales. Saliendo.")
        exit()

    print("\n--- OPTIMIZACIÓN DE PARÁMETROS ---")
    best_params = optimize_parameters(signals)
    if best_params is None:
        print("⚠️ Usando parámetros por defecto: tp_atr=3, sl_atr=2, atr_window=14, pidelta_window=13")
        best_params = {'tp_atr': 3, 'sl_atr': 2, 'atr_window': 14, 'pidelta_window': 13, 'tension_quantile': 0.8}

    analyze_top_signals(signals, best_params)

    print("\n✅ Proceso completado. Archivos: 'escaneo_filtrado.csv', 'optimization_results.csv'")
