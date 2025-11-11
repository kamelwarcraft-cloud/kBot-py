# -*- coding: utf-8 -*-
"""
kBot Trading — Intraday (5m) + Daily (1d) avec réactions quasi-instant
- Poll Telegram ~1.5s pour lire les réactions (✅/❌) et confirmer immédiatement
- Signaux tagués: [RSI Safe] / [RSI 2] pour 5m et 1d
- Suggestion de TP (TP1/TP2/TP3) selon la force du signal (écart RSI + tendance)
- N'envoie des notifs que s'il y a un signal BUY/SELL
- Boutons Telegram + fallback Discord si Telegram échoue
- Récap horaire Intraday & Daily (TP/SL / positions / signaux) avec type RSI affiché
- Yahoo Finance fallback + état persistant dans state.json
"""

import os, re, math, json, requests, datetime as dt, warnings, time, uuid
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)

# ========== .env ==========
load_dotenv()

# ==================== CONFIG ====================
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL").split(",") if s.strip()]
INTERVAL = os.getenv("INTERVAL", "5m")
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "30"))
RISK_PROFILE = int(os.getenv("RISK_PROFILE", "2"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
EQUITY = float(os.getenv("EQUITY", "10000"))
TZ = os.getenv("TIMEZONE", "Europe/Paris")

# News (AlphaVantage)
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
USE_NEWS = os.getenv("USE_NEWS", "false").lower() == "true"

# Telegram / Discord
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")
DC_HOOK = os.getenv("DISCORD_WEBHOOK_URL")
TELEGRAM_USE_REACTIONS = os.getenv("TELEGRAM_USE_REACTIONS", "true").lower() == "true"
TELEGRAM_DEBUG = os.getenv("TELEGRAM_DEBUG", "false").lower() == "true"

# Alpaca (optionnel)
ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
SEND_ORDERS = os.getenv("SEND_ORDERS", "false").lower() == "true"

# Déduplication & re-notif
DEDUPE_MINUTES = int(os.getenv("DEDUPE_MINUTES", "10"))
NOTIFY_DUP_IF_MOVE_ATR = float(os.getenv("NOTIFY_DUP_IF_MOVE_ATR", "0.3"))

# Pré/post-market
INCLUDE_PREPOST = os.getenv("INCLUDE_PREPOST", "true").lower() == "true"

# Option : accepter trend neutre (plus de signaux, plus risqué)
ALLOW_TREND_NEUTRAL = os.getenv("ALLOW_TREND_NEUTRAL", "false").lower() == "true"

# Debug “pourquoi pas de signal ?”
DEBUG_NO_SIGNAL = os.getenv("DEBUG_NO_SIGNAL", "false").lower() == "true"

STATE_FILE = os.getenv("STATE_FILE", "state.json")

def now_paris():
    return dt.datetime.now(pytz.timezone(TZ))

PROFILE = {
    1: {"k": 1.0, "tp": [0.8, 1.2, 2.0]},
    2: {"k": 1.5, "tp": [1.0, 2.0, 3.0]},
    3: {"k": 2.0, "tp": [1.5, 3.0, 4.0]},
}[max(1, min(3, RISK_PROFILE))]

# ==================== STATE ====================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "open_trades": [],
            "recent_signals": [],
            "last_update_id": 0,
            "last_recap_hour_intraday": None,
            "last_recap_hour_daily": None,
        }

def save_state(st):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)

# ==================== NOTIFY ====================
def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")

def _escape_angle_brackets_for_html(text: str) -> str:
    """
    Protège des séquences numériques comme '39.2<40' ou '62>55' qui cassent le parseur HTML Telegram.
    Remplace par '≤' et '≥' uniquement quand il s'agit de comparaisons entre nombres.
    """
    text = re.sub(r'(?<=\d)\s*<\s*(?=\d)', ' ≤ ', text)
    text = re.sub(r'(?<=\d)\s*>\s*(?=\d)', ' ≥ ', text)
    return text

def tg_send_message(text: str, reply_markup: dict | None = None) -> dict | None:
    if not (TG_TOKEN and TG_CHAT):
        print("[TG] Config manquante (TOKEN/CHAT_ID). Message:\n", text)
        return None
    try:
        text = _escape_angle_brackets_for_html(text)
        payload = {
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json=payload, timeout=12,
        )
        if TELEGRAM_DEBUG:
            print("[TG] sendMessage status:", r.status_code)
            try:
                print("[TG] response:", r.json())
            except Exception:
                print("[TG] raw:", r.text)

        if r.status_code != 200:
            print("[TG] ERREUR HTTP:", r.status_code, r.text[:500])
            return None

        data = r.json()
        if not data.get("ok", False):
            print("[TG] ERREUR API:", data.get("description", "inconnu"))
            return None
        return data
    except Exception as e:
        print("[TG] Exception:", str(e))
        return None

def notify(text: str, reply_markup: dict | None = None):
    ok = False
    resp = tg_send_message(text, reply_markup=reply_markup)
    ok = bool(resp)
    if DC_HOOK and not ok:
        try:
            requests.post(DC_HOOK, json={"content": _strip_html(text)}, timeout=10)
            ok = True
        except Exception:
            pass
    if not ok:
        print("[NOTIF STDOUT]\n", text)

def tg_set_reactions(chat_id: str, message_id: int, emojis: list[str], chat_type: Optional[str]):
    # Reactions Bot API non supportées en "private".
    if chat_type not in ("group", "supergroup", "channel"):
        return
    if not (TG_TOKEN and chat_id and message_id and emojis):
        return
    try:
        reaction = [{"type": "emoji", "emoji": e} for e in emojis]
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/setMessageReaction",
            json={"chat_id": chat_id, "message_id": message_id, "reaction": reaction, "is_big": False},
            timeout=10
        )
        if TELEGRAM_DEBUG:
            print("[TG] setMessageReaction:", r.status_code, r.text[:300])
    except Exception as e:
        if TELEGRAM_DEBUG:
            print("[TG] setMessageReaction exception:", str(e))

def send_signal_with_buttons(text: str, trade_id: str) -> Optional[int]:
    kb = {"inline_keyboard": [[
        {"text": "✅ Je suis entré (V)", "callback_data": f"CONFIRM|{trade_id}"},
        {"text": "❌ Ignorer (X)",      "callback_data": f"IGNORE|{trade_id}"},
    ]]}
    resp = tg_send_message(text, reply_markup=kb)

    msg_id = None
    chat_type = None
    if isinstance(resp, dict):
        res = resp.get("result") or resp.get("message")
        if isinstance(res, dict):
            msg_id = res.get("message_id")
            chat = res.get("chat") or {}
            chat_type = chat.get("type")

    if TELEGRAM_USE_REACTIONS and msg_id:
        tg_set_reactions(TG_CHAT, msg_id, ["✅", "❌"], chat_type)

    if DC_HOOK and (msg_id is None):
        try:
            requests.post(DC_HOOK, json={"content": _strip_html(text)}, timeout=10)
        except Exception:
            pass

    return msg_id

def poll_telegram_updates(state: dict):
    """Lit boutons & (si dispo) réactions. Appelée ~chaque 1.5s."""
    if not TG_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates"
        params = {"timeout": 0, "allowed_updates": json.dumps(["callback_query","message_reaction"])}
        if state.get("last_update_id"):
            params["offset"] = state["last_update_id"] + 1
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for upd in data.get("result", []):
            state["last_update_id"] = max(state.get("last_update_id", 0), upd.get("update_id", 0))

            cq = upd.get("callback_query")
            if cq and "data" in cq:
                cd = cq["data"]
                if "|" in cd:
                    action, trade_id = cd.split("|", 1)
                    handle_callback_action(state, action, trade_id)

            mr = upd.get("message_reaction")
            if mr and TELEGRAM_USE_REACTIONS:
                chat = mr.get("chat") or {}
                chat_id = str(chat.get("id",""))
                message_id = mr.get("message_id")
                new = mr.get("new_reaction") or []
                last_emoji = None
                for item in new:
                    if item.get("type") == "emoji" and "emoji" in item:
                        last_emoji = item["emoji"]
                if chat_id and message_id and last_emoji:
                    apply_reaction_to_trade(state, chat_id, message_id, last_emoji)
    except Exception as e:
        if TELEGRAM_DEBUG:
            print("[TG] getUpdates exception:", str(e))

def handle_callback_action(state: dict, action: str, trade_id: str):
    for tr in state["open_trades"]:
        if tr.get("id") == trade_id:
            if action == "CONFIRM":
                tr["user_confirmed"] = True
                notify(f"✅ Confirmation reçue pour {tr['symbol']} ({tr['side']}) — trade {tr['id'][:8]}")
            elif action == "IGNORE":
                tr["user_confirmed"] = False
                notify(f"❌ Signal ignoré pour {tr['symbol']} ({tr['side']}) — trade {tr['id'][:8]}")
            save_state(state)
            break

def apply_reaction_to_trade(state: dict, chat_id: str, message_id: int, emoji: str):
    for tr in state["open_trades"]:
        if tr.get("message_id") == message_id:
            if emoji == "✅":
                tr["user_confirmed"] = True
                notify(f"✅ Confirmation reçue pour {tr['symbol']} ({tr['side']}) — trade {tr['id'][:8]}")
            elif emoji == "❌":
                tr["user_confirmed"] = False
                notify(f"❌ Signal ignoré pour {tr['symbol']} ({tr['side']}) — trade {tr['id'][:8]}")
            save_state(state)
            break

# ==================== INDICATEURS ====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# ==================== DATA (Yahoo fallback) ====================
def yfdl(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    pd_str = f"{period_days}d"
    try_order = [(pd_str, interval)]
    if interval.endswith("m"):
        try_order.append(("15d", interval))
    else:
        try_order.append(("90d", "1d"))

    for per, itv in try_order:
        try:
            df = yf.download(
                symbol, period=per, interval=itv, auto_adjust=True,
                progress=False, prepost=INCLUDE_PREPOST, threads=False
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()

def get_ohlc(symbol: str, period_days: int, interval: str) -> pd.DataFrame:
    df = yfdl(symbol, period_days, interval)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().copy()
    df["ema50"] = ema(df["Close"], 50)
    df["ema200"] = ema(df["Close"], 200)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["Close"])
    df["rsi"] = rsi(df["Close"], 14)
    df["atr"] = atr(df["High"], df["Low"], df["Close"], 14)
    return df.dropna()

def get_ohlc_5m(symbol: str) -> pd.DataFrame:
    return get_ohlc(symbol, LOOKBACK_DAYS, INTERVAL)

def get_ohlc_daily(symbol: str) -> pd.DataFrame:
    return get_ohlc(symbol, 180, "1d")

def fetch_last_price(symbol: str) -> Optional[float]:
    try:
        df = yf.download(symbol, period="1d", interval="1m", auto_adjust=True, progress=False, prepost=INCLUDE_PREPOST)
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

def news_bias(symbol: str) -> float:
    if not (USE_NEWS and ALPHAVANTAGE_API_KEY):
        return 0.0
    try:
        url = ("https://www.alphavantage.co/query"
               f"?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHAVANTAGE_API_KEY}")
        r = requests.get(url, timeout=10).json()
        items = r.get("feed", [])[:10]
        scores = [float(it.get("overall_sentiment_score", 0)) for it in items if it.get("overall_sentiment_score") is not None]
        if not scores:
            return 0.0
        return float(np.clip(np.mean(scores), -1, 1))
    except Exception:
        return 0.0

# ==================== DEBUG AIDE ====================
def _debug_last_candle_report(symbol: str, df, buy_rsi, sell_rsi, timeframe: str):
    if not DEBUG_NO_SIGNAL or df is None or df.empty:
        return None
    try:
        c = float(df["Close"].iloc[-1]); ema50=float(df["ema50"].iloc[-1]); ema200=float(df["ema200"].iloc[-1])
        macd_line=float(df["macd"].iloc[-1]); rsi_v=float(df["rsi"].iloc[-1])
        trend_up = (ema50>ema200) and (macd_line>0)
        trend_dn = (ema50<ema200) and (macd_line<0)

        buy_ok  = (trend_up or ALLOW_TREND_NEUTRAL) and (rsi_v < buy_rsi)
        sell_ok = (trend_dn or ALLOW_TREND_NEUTRAL) and (rsi_v > sell_rsi)

        need = []
        if not trend_up and not trend_dn:
            need.append("trend neutre (attend ↑ ou ↓)")
        if trend_up and not (rsi_v < buy_rsi):
            need.append(f"BUY KO: RSI {rsi_v:.1f} (doit être < {buy_rsi})")
        if trend_dn and not (rsi_v > sell_rsi):
            need.append(f"SELL KO: RSI {rsi_v:.1f} (doit être > {sell_rsi})")

        status = "OK BUY" if buy_ok else ("OK SELL" if sell_ok else "AUCUN")
        need_str = ", ".join(need) if need else "—"
        return f"[{timeframe}] {symbol}: {status} | C={c:.2f}, RSI={rsi_v:.1f}, MACD={macd_line:.3f} | Manque: {need_str}"
    except Exception:
        return None

# ==================== SIGNAL LOGIC ====================
@dataclass
class Signal:
    id: str
    symbol: str
    side: str
    price: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    size: int
    rationale: str
    news: float
    timeframe: str  # "5m" ou "1d"
    rsi_mode: str   # "RSI Safe" ou "RSI 2"
    tp_pick: int    # 1/2/3
    tp_reason: str
    created_at: str # ISO

def position_size(entry: float, sl: float) -> int:
    risk_money = EQUITY * RISK_PER_TRADE
    per_share_risk = abs(entry - sl)
    if per_share_risk <= 0:
        return 0
    shares = math.floor(risk_money / per_share_risk)
    return max(0, shares)

def _interesting_duplicate(prev_trade: dict, entry: float, atr_value: float) -> bool:
    if NOTIFY_DUP_IF_MOVE_ATR <= 0:
        return False
    move = abs(entry - prev_trade.get("entry", entry))
    return (atr_value > 0) and (move >= NOTIFY_DUP_IF_MOVE_ATR * atr_value)

def _suggest_tp(side: str, rsi_val: float, buy_thresh: float, sell_thresh: float, trend_ok: bool) -> tuple[int, str]:
    """
    Heuristique simple :
    - Ecart RSI au seuil < 3  ⇒ TP1 (conservateur)
    - Ecart entre 3 et 8      ⇒ TP2 (moyen)
    - Ecart >= 8 & trend OK   ⇒ TP3 (agressif) sinon TP2
    """
    if side == "BUY":
        gap = buy_thresh - rsi_val  # positif si rsi < seuil (meilleur)
    else:
        gap = rsi_val - sell_thresh # positif si rsi > seuil (meilleur)

    if gap < 3:
        return 1, f"Écart RSI faible ({gap:.1f}), TP1"
    if gap < 8:
        return 2, f"Écart RSI moyen ({gap:.1f}), TP2"
    if trend_ok:
        return 3, f"Écart RSI fort ({gap:.1f}) + trend OK, TP3"
    return 2, f"Écart RSI fort ({gap:.1f}) mais trend neutre/faible, TP2"

def _signal_from_df(symbol: str, df: pd.DataFrame, buy_rsi_thresh: float, sell_rsi_thresh: float, timeframe: str, rsi_mode: str) -> Optional[Signal]:
    if df is None or df.empty:
        return None

    price     = float(df["Close"].iloc[-1])
    ema50     = float(df["ema50"].iloc[-1])
    ema200    = float(df["ema200"].iloc[-1])
    macd_line = float(df["macd"].iloc[-1])
    rsi_v     = float(df["rsi"].iloc[-1])
    atr_v     = float(df["atr"].iloc[-1])

    trend_up = (ema50 > ema200) and (macd_line > 0)
    trend_down = (ema50 < ema200) and (macd_line < 0)
    bias_news = news_bias(symbol) if USE_NEWS else 0.0
    R = PROFILE["k"] * atr_v

    buy_ok  = ((trend_up or ALLOW_TREND_NEUTRAL)  and rsi_v < buy_rsi_thresh and (bias_news >= -0.3))
    sell_ok = ((trend_down or ALLOW_TREND_NEUTRAL) and rsi_v > sell_rsi_thresh and (bias_news <= +0.3))

    if buy_ok:
        sl = price - R
        tps = [price + m * R for m in PROFILE["tp"]]
        size = position_size(price, sl)
        if size > 0:
            tp_pick, tp_reason = _suggest_tp("BUY", rsi_v, buy_rsi_thresh, sell_rsi_thresh, trend_up)
            return Signal(
                id=str(uuid.uuid4()),
                symbol=symbol, side="BUY", price=price, sl=sl,
                tp1=tps[0], tp2=tps[1], tp3=tps[2], size=size,
                rationale=f"Trend {'↑' if trend_up else '·'} & MACD {macd_line:+.3f}, RSI {rsi_v:.1f}≤{buy_rsi_thresh}, news {bias_news:+.2f}",
                news=bias_news, timeframe=timeframe, rsi_mode=rsi_mode,
                tp_pick=tp_pick, tp_reason=tp_reason,
                created_at=now_paris().isoformat()
            )

    if sell_ok:
        sl = price + R
        tps = [price - m * R for m in PROFILE["tp"]]
        size = position_size(price, sl)
        if size > 0:
            tp_pick, tp_reason = _suggest_tp("SELL", rsi_v, buy_rsi_thresh, sell_rsi_thresh, trend_down)
            return Signal(
                id=str(uuid.uuid4()),
                symbol=symbol, side="SELL", price=price, sl=sl,
                tp1=tps[0], tp2=tps[1], tp3=tps[2], size=size,
                rationale=f"Trend {'↓' if trend_down else '·'} & MACD {macd_line:+.3f}, RSI {rsi_v:.1f}≥{sell_rsi_thresh}, news {bias_news:+.2f}",
                news=bias_news, timeframe=timeframe, rsi_mode=rsi_mode,
                tp_pick=tp_pick, tp_reason=tp_reason,
                created_at=now_paris().isoformat()
            )
    return None

def gen_5m_safe(symbol):
    df = get_ohlc_5m(symbol)
    sig = _signal_from_df(symbol, df, 35, 65, "5m", "RSI Safe")
    return sig

def gen_5m_relaxed(symbol):
    df = get_ohlc_5m(symbol)
    sig = _signal_from_df(symbol, df, 40, 60, "5m", "RSI 2")
    return sig

def gen_1d_safe(symbol):
    df = get_ohlc_daily(symbol)
    sig = _signal_from_df(symbol, df, 35, 65, "1d", "RSI Safe")
    return sig

def gen_1d_relaxed(symbol):
    df = get_ohlc_daily(symbol)
    sig = _signal_from_df(symbol, df, 40, 60, "1d", "RSI 2")
    return sig

# ==================== OPEN TRADE TRACKING ====================
def track_new_signal(state: dict, sig: Signal) -> bool:
    """
    Ajoute le trade si ce n'est pas un doublon récent (même symbole/side/timeframe/rsi_mode).
    Fenêtre: DEDUPE_MINUTES.
    Autorise re-notif si écart prix >= NOTIFY_DUP_IF_MOVE_ATR * ATR (si activé).
    """
    now_ts = pd.Timestamp(sig.created_at)
    for t in state["open_trades"]:
        if t["symbol"] == sig.symbol and t["timeframe"] == sig.timeframe and t["side"] == sig.side and t.get("rsi_mode")==sig.rsi_mode:
            dt_sec = abs(pd.Timestamp(t["created_at"]) - now_ts).total_seconds()
            if dt_sec < 60 * DEDUPE_MINUTES:
                atr_value = abs(sig.price - sig.sl) / max(PROFILE["k"], 1e-9)
                if _interesting_duplicate(t, sig.price, atr_value):
                    break
                return False

    state["open_trades"].append({
        "id": sig.id,
        "symbol": sig.symbol,
        "side": sig.side,
        "entry": sig.price,
        "sl": sig.sl,
        "tp1": sig.tp1,
        "tp2": sig.tp2,
        "tp3": sig.tp3,
        "size": sig.size,
        "timeframe": sig.timeframe,
        "rsi_mode": sig.rsi_mode,
        "tp_pick": sig.tp_pick,
        "tp_reason": sig.tp_reason,
        "created_at": sig.created_at,
        "user_confirmed": None,
        "hits": {"sl": False, "tp1": False, "tp2": False, "tp3": False},
        "message_id": None,
    })
    return True

def attach_message_id(state: dict, sig_id: str, message_id: int):
    for tr in state["open_trades"]:
        if tr["id"] == sig_id:
            tr["message_id"] = message_id
            break
    save_state(state)

def check_hits_for_trade(tr: dict):
    price = fetch_last_price(tr["symbol"])
    if price is None:
        return

    if tr["side"] == "BUY":
        if not tr["hits"]["tp1"] and price >= tr["tp1"]:
            tr["hits"]["tp1"] = True
        if not tr["hits"]["tp2"] and price >= tr["tp2"]:
            tr["hits"]["tp2"] = True
        if not tr["hits"]["tp3"] and price >= tr["tp3"]:
            tr["hits"]["tp3"] = True
        if not tr["hits"]["sl"] and price <= tr["sl"]:
            tr["hits"]["sl"] = True
    else:  # SELL
        if not tr["hits"]["tp1"] and price <= tr["tp1"]:
            tr["hits"]["tp1"] = True
        if not tr["hits"]["tp2"] and price <= tr["tp2"]:
            tr["hits"]["tp2"] = True
        if not tr["hits"]["tp3"] and price <= tr["tp3"]:
            tr["hits"]["tp3"] = True
        if not tr["hits"]["sl"] and price >= tr["sl"]:
            tr["hits"]["sl"] = True

# ==================== HOURLY RECAP ====================
def hourly_recap(state: dict, timeframe: str):
    now = now_paris()
    hour_key = now.strftime("%Y-%m-%d %H")
    keyname = "last_recap_hour_intraday" if timeframe=="5m" else "last_recap_hour_daily"
    if state.get(keyname) == hour_key:
        return

    open_trades = [t for t in state["open_trades"] if t["timeframe"] == timeframe]
    for tr in open_trades:
        try:
            check_hits_for_trade(tr)
        except Exception:
            pass

    lines = []
    title = "🕑 Récap horaire (Intraday)" if timeframe=="5m" else "🗓️ Récap horaire (Daily)"
    lines.append(f"{title} — {now.strftime('%Y-%m-%d %H:00')}")

    # SL/TP
    lines.append("— SL/TP atteints —")
    any_hits = False
    for tr in open_trades:
        h = tr["hits"]
        parts = []
        if h["sl"]: parts.append("SL")
        if h["tp1"]: parts.append("TP1")
        if h["tp2"]: parts.append("TP2")
        if h["tp3"]: parts.append("TP3")
        if parts:
            any_hits = True
            lines.append(f"• [{tr.get('rsi_mode','?')}] {tr['symbol']} ({tr['side']}) — {', '.join(parts)}")
    if not any_hits:
        lines.append("Aucun")

    # Entrées confirmées
    lines.append("— Entrées confirmées —")
    confirmed = [t for t in open_trades if t["user_confirmed"] is True]
    if confirmed:
        for tr in confirmed:
            lines.append(f"• [{tr.get('rsi_mode','?')}] {tr['symbol']} ({tr['side']}) @ {tr['entry']:.2f} | SL {tr['sl']:.2f} | TP {tr['tp1']:.2f}/{tr['tp2']:.2f}/{tr['tp3']:.2f} | TP suggéré: TP{tr.get('tp_pick','?')}")
    else:
        lines.append("Aucune")

    # Positions suivies
    lines.append("— Achats/Ventes en cours —")
    if open_trades:
        for tr in open_trades:
            status = "✅" if tr["user_confirmed"] else ("⏳" if tr["user_confirmed"] is None else "❌")
            lines.append(f"• [{tr.get('rsi_mode','?')}] {status} {tr['symbol']} {tr['side']} @ {tr['entry']:.2f} | SL {tr['sl']:.2f} | TP {tr['tp1']:.2f}/{tr['tp2']:.2f}/{tr['tp3']:.2f} | TP suggéré: TP{tr.get('tp_pick','?')}")
    else:
        lines.append("Aucun")

    # Signaux de la dernière heure
    lines.append("— Signaux (dernière heure) —")
    cutoff = now - dt.timedelta(hours=1)
    sigs = [s for s in state.get("recent_signals", []) if s["timeframe"]==timeframe and pd.Timestamp(s["created_at"])>=cutoff]
    if sigs:
        for s in sigs:
            t_local = "?"
            try:
                ts = pd.Timestamp(s["created_at"])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC").tz_convert(TZ)
                else:
                    ts = ts.tz_convert(TZ)
                t_local = ts.strftime("%H:%M")
            except Exception:
                pass
            lines.append(f"• [{s.get('rsi_mode','?')}] {s['symbol']} {s['side']} @ {s['price']:.2f} ({t_local}) | TP suggéré: TP{s.get('tp_pick','?')}")
    else:
        lines.append("Aucun signal détecté")

    notify("\n".join(lines))
    state[keyname] = hour_key
    save_state(state)

# ==================== MAIN ====================
def fmt_price(x: float) -> str:
    return f"{x:.2f}"

def notify_signal(sig: Signal) -> Optional[int]:
    # Titres explicites avec type RSI
    scope = "5m" if sig.timeframe=="5m" else "1d"
    title = f"📈 Signal ({scope}) — [{sig.rsi_mode}]"
    msg = (
        f"{title}\n"
        f"<b>{sig.symbol}</b> — <b>{sig.side}</b>\n"
        f"Entrée: {fmt_price(sig.price)} | SL: {fmt_price(sig.sl)}\n"
        f"TP: {fmt_price(sig.tp1)} / {fmt_price(sig.tp2)} / {fmt_price(sig.tp3)}\n"
        f"Taille: {sig.size}\n"
        f"TP suggéré: <b>TP{sig.tp_pick}</b> ({sig.tp_reason})\n"
        f"Actu (score): {sig.news:+.2f}\n"
        f"Raison: {sig.rationale}\n"
        f"id: <code>{sig.id[:8]}</code>\n"
    )

    msg_id = None
    if TG_TOKEN and TG_CHAT:
        msg_id = send_signal_with_buttons(msg, sig.id)

    if DC_HOOK and (msg_id is None):
        try:
            requests.post(DC_HOOK, json={"content": _strip_html(msg)}, timeout=10)
        except Exception:
            pass

    return msg_id

def generate_signals(symbols: List[str]) -> List[Signal]:
    out: List[Signal] = []
    for sym in symbols:
        for gen in (gen_5m_safe, gen_5m_relaxed, gen_1d_safe, gen_1d_relaxed):
            try:
                s = gen(sym)
                if s:
                    out.append(s)
            except Exception:
                pass
    return out

def main_once(state: dict):
    """Exécute une analyse et envoie les signaux; met à jour récents + open_trades."""
    new_signals = generate_signals(SYMBOLS)

    if new_signals:
        for sig in new_signals:
            if track_new_signal(state, sig):
                mid = notify_signal(sig)
                if mid:
                    attach_message_id(state, sig.id, mid)
            state["recent_signals"].append({
                "id": sig.id, "symbol": sig.symbol, "side": sig.side,
                "price": sig.price, "timeframe": sig.timeframe, "rsi_mode": sig.rsi_mode,
                "tp_pick": sig.tp_pick, "created_at": sig.created_at
            })
        # garder 24h de signaux récents
        cutoff = now_paris() - dt.timedelta(hours=24)
        state["recent_signals"] = [s for s in state["recent_signals"] if pd.Timestamp(s["created_at"]) >= cutoff]
        save_state(state)

def run_loop():
    state = load_state()

    # Ping Telegram au lancement (diagnostic)
    if TG_TOKEN and TG_CHAT:
        tg_send_message("🚦 kBot démarré — réactions quasi-instantanées activées (boutons ✅/❌).")

    # Timers
    next_analysis = time.time()          # analyse dès le démarrage
    analysis_period = 300.0              # 5 minutes
    last_hour_intraday = None
    last_hour_daily = None

    while True:
        # 1) Poll Telegram très fréquemment pour réactions instant
        poll_telegram_updates(state)

        # 2) Analyse toutes les 5 minutes
        now_t = time.time()
        if now_t >= next_analysis:
            main_once(state)
            next_analysis = now_t + analysis_period

        # 3) Récaps horaires (une fois par heure)
        now = now_paris()
        hour_key = now.strftime("%Y-%m-%d %H")
        if last_hour_intraday != hour_key:
            try:
                hourly_recap(state, "5m")
            except Exception:
                pass
            last_hour_intraday = hour_key

        if last_hour_daily != hour_key:
            try:
                hourly_recap(state, "1d")
            except Exception:
                pass
            last_hour_daily = hour_key

        # Petite sieste pour ne pas surcharger le CPU et lire vite les réactions
        time.sleep(1.5)

if __name__ == "__main__":
    run_loop()
