# tradezella_clone_singlefile_final.py
# Single-file Streamlit app — regenerated with universal importer/normalizer
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from supabase import create_client, Client

# ---------------- Supabase Setup ----------------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- Authentication ----------------

def login():
    st.subheader("Login")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if not email or not password:
            st.error("Please enter both email and password")
            return

        try:
            res = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if res.user is None:
                st.error("Login failed: Please confirm your email first")
                return

            # 🔥 THIS IS THE MISSING PIECE
            supabase.auth.set_session(
                res.session.access_token,
                res.session.refresh_token
            )

            st.session_state.user = res.user
            st.success("Logged in successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Invalid credentials: {e}")
            
# ---------------- Invite-Only Signup with Duplicate Email Check ----------------
def signup():
    st.subheader("Sign Up (Invite Only)")

    # User inputs
    display_name = st.text_input("Your Name", key="signup_name")
    email = st.text_input("New Email", key="signup_email")
    password = st.text_input("New Password", type="password", key="signup_password")
    invite_code = st.text_input("Invite Code", key="signup_invite")

    if st.button("Create Account"):

        # ---------------- Basic validation ----------------
        if not display_name or not email or not password or not invite_code:
            st.error("All fields are required")
            return

        if len(password) < 6:
            st.error("Password must be at least 6 characters")
            return

        invite_code_clean = invite_code.strip().upper()

        # ---------------- Check Invite Code ----------------
        invite_check = supabase.table("invite_codes") \
            .select("*") \
            .eq("code", invite_code_clean) \
            .eq("is_active", True) \
            .execute()

        if not invite_check.data:
            st.error("Invalid invite code")
            return

        invite = invite_check.data[0]

        # Check usage limits
        if invite["max_uses"] is not None and invite["uses_count"] >= invite["max_uses"]:
            st.error("Invite code has reached its usage limit")
            return

        # ---------------- Check If Email Is Banned ----------------
        ban_check = supabase.table("banned_users") \
            .select("*") \
            .eq("email", email) \
            .execute()

        if ban_check.data:
            st.error("This email has been banned.")
            return

        # ---------------- Check If Email Already Exists ----------------
        try:
            existing_user = supabase.auth.admin.get_user_by_email(email)
            if existing_user.user is not None:
                st.error("Account with this email already exists. Use login or reset password.")
                return
        except Exception:
            # If error occurs, assume user does not exist
            pass

        # ---------------- Create User ----------------
        try:
            res = supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": {"full_name": display_name}}
            })

            # Handle signup errors
            if hasattr(res, "error") and res.error:
                st.error(res.error.message)
                return

            if res.user is None:
                st.info("Check your email to confirm your account.")
                return

            user_id = res.user.id

            # ---------------- Update Invite Usage ----------------
            supabase.table("invite_codes") \
                .update({"uses_count": invite["uses_count"] + 1}) \
                .eq("id", invite["id"]) \
                .execute()

            # ---------------- Create Profile ----------------
            # Ensure it matches your table schema (id = user.id)
            supabase.table("profiles").insert({
                "id": user_id,
                "full_name": display_name,
                "subscription_tier": "free"
            }).execute()

            st.success("Account created successfully! Please confirm your email.")
            st.session_state.user = res.user
            st.rerun()

        except Exception as e:
            st.error(f"Signup failed: {e}")
# ---------------- Protect App ----------------

if st.session_state.user is None:
    choice = st.radio("Choose Option", ["Login", "Sign Up"])

    if choice == "Login":
        login()
    else:
        signup()

    st.stop()

# ---------------- Page config ----------------
st.set_page_config(page_title="TradeZella — Pixel-Perfect Clone (Final)", layout="wide")

# ---------------- CSS (pixel-tuned) ----------------
st.markdown(
    """
    <style>
    :root{--bg:#0f1724;--card:#0b1220;--muted:#94a3b8;--accent-green:#26a269;--accent-red:#ff5b5b;--muted-border:rgba(255,255,255,0.03);}    
    html, body, .reportview-container, .main, .block-container {background-color: var(--bg);color: #e6eef8;font-family: Inter, sans-serif;}
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:14px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); border: 1px solid var(--muted-border); }
    .kpi-title { font-size:12px; color:var(--muted); margin-bottom:6px; }
    .kpi-val { font-size:20px; font-weight:700; color:#fff; display:flex; align-items:center; justify-content:space-between; }
    .ring-sz { width:48px; height:48px; }
    .awl-container { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .awl-side { width:72px; display:flex; flex-direction:column; align-items:center; }
    .circle-small { width:36px; height:36px; border-radius:999px; display:flex; align-items:center; justify-content:center; }
    .circle-green { background: linear-gradient(180deg, rgba(38,162,105,0.16), rgba(38,162,105,0.06)); border:2px solid rgba(38,162,105,0.18); color:var(--accent-green); font-weight:700; }
    .circle-red { background: linear-gradient(180deg, rgba(255,91,91,0.12), rgba(255,91,91,0.04)); border:2px solid rgba(255,91,91,0.12); color:var(--accent-red); font-weight:700; }
    .awl-num { margin-top:6px; font-weight:700; }
    /* Calendar: 7 day columns + 1 weekly summary column */
    .calendar-grid { display:grid; grid-template-columns:repeat(8,1fr); gap:6px; margin-top:6px; }
    .calendar-day { border-radius:8px; padding:8px; text-align:center; font-size:12px; min-height:68px; display:flex; flex-direction:column; justify-content:center; align-items:center; }
    .day-num { font-weight:700; font-size:14px; margin-bottom:6px; }
    .positive { background: linear-gradient(180deg, rgba(38,162,105,0.12), rgba(38,162,105,0.06)); color: var(--accent-green); border:1px solid rgba(38,162,105,0.15); }
    .negative { background: linear-gradient(180deg, rgba(255,91,91,0.08), rgba(255,91,91,0.04)); color: var(--accent-red); border:1px solid rgba(255,91,91,0.12); }
    .neutral { background: #071021; color: #9ca3af; border:1px solid rgba(255,255,255,0.02); }
    /* Weekly summary column styling: distinct but keeps green/red tone */
    .weekly-summary { background: linear-gradient(180deg, rgba(14,21,40,0.6), rgba(11,17,32,0.6)); border: 1px solid rgba(255,255,255,0.04); min-height:92px; font-size:13px; display:flex; flex-direction:column; justify-content:center; align-items:center; padding:10px; }
    .weekly-summary.positive { background: linear-gradient(180deg, rgba(38,162,105,0.12), rgba(38,162,105,0.06)); color: var(--accent-green); border:1px solid rgba(38,162,105,0.15); }
    .weekly-summary.negative { background: linear-gradient(180deg, rgba(255,91,91,0.08), rgba(255,91,91,0.04)); color: var(--accent-red); border:1px solid rgba(255,91,91,0.12); }
    .weekly-summary.neutral { background: #081022; color: #9ca3af; border:1px solid rgba(255,255,255,0.02); }
    /* thin white separator between day columns and weekly summary column */
    .weekly-summary { border-left: 1px solid rgba(255,255,255,0.06); }
    .weekly-summary .day-num { font-size:16px; font-weight:800; margin-bottom:6px; }
    .side-buttons { display:flex; flex-direction:column; gap:8px; padding-bottom:8px; }
    .side-label { font-size:13px; color:#e6eef8; margin-left:6px; }
    table.smalltbl { width:100%; border-collapse:collapse; font-size:13px; }
    table.smalltbl th { text-align:center; color:var(--muted); padding-bottom:6px; }
    table.smalltbl td { padding:8px 6px; text-align:center; }
    .css-1d391kg { padding-top: 8px; }
    .insight-card { padding:10px; border-radius:8px; background:#09101a; border:1px solid rgba(255,255,255,0.02); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Utilities ----------------
def parse_time(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x)
    for fmt in ("%Y-%m-%d %H:%M:%S","%Y.%m.%d %H:%M","%d.%m.%Y %H:%M:%S","%Y-%m-%dT%H:%M:%S","%Y-%m-%d","%m/%d/%Y %H:%M:%S","%m/%d/%Y"):
        try:
            return pd.to_datetime(x, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT

def process_mt5_df(df: pd.DataFrame):
    """
    Robust processing for many broker CSV/XLSX formats.

    - Maps common alternative column names to the expected internal names.
    - Parses times, numeric columns.
    - Groups rows by Ticket/Position ID, computing EntryPrice/ExitPrice, Swap, Profit, etc.
    - If Profit is missing but EntryPrice & ExitPrice exist, computes a fallback Profit=(ExitPrice-EntryPrice)*Lots.
    """
    df = df.copy()
    df = df.rename(columns=lambda c: str(c).strip())

    # Common aliases for the fields we need
    aliases = {
        "Ticket": ["ticket", "position id", "positionid", "posid", "order", "order id", "orderid", "id"],
        "Symbol": ["symbol", "instrument", "ticker", "pair"],
        "Time": ["time", "open time", "opentime", "open_date", "open date", "date", "created at", "close date", "close_date", "close time"],
        "Action": ["action", "type", "side"],
        "Lots": ["lots", "volume", "size", "qty", "quantity"],
        "Price": ["price", "open price", "openprice", "entryprice", "entry price"],
        "ExitPrice": ["close price", "closeprice", "current price", "currentprice", "exitprice", "close", "close_price"],
        "Swap": ["swap", "commission", "commissions"],
        "Profit": ["profit", "pnl", "pl", "profit/loss", "profit (usd)", "net", "net profit"]
    }

    # create reverse lookup mapping existing col -> normalized name
    col_map = {}
    existing_cols = [c for c in df.columns]
    for target, keys in aliases.items():
        for k in keys:
            for c in existing_cols:
                if c.lower() == k.lower():
                    col_map[c] = target
                    break
            if any(v == target for v in col_map.values()):
                break

    # Apply mapping to create normalized column names
    df = df.rename(columns=lambda c: col_map.get(c, c))

    # Ensure required columns exist
    for c in ["Ticket", "Symbol", "Time", "Action", "Lots", "Price", "ExitPrice", "Swap", "Profit"]:
        if c not in df.columns:
            df[c] = np.nan

    # Parse times
    if "Time" in df.columns:
        df["Time"] = df["Time"].apply(parse_time)
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    else:
        df["Time"] = pd.NaT

    # Numeric conversion
    for c in ["Profit", "Lots", "Price", "Swap", "ExitPrice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Normalize action strings
    df["Action"] = df["Action"].astype(str).str.strip().str.upper().replace({"B":"BUY","S":"SELL","LONG":"BUY","SHORT":"SELL"})

    # Fill Ticket if missing
    if df["Ticket"].isna().any():
        mask = df["Ticket"].isna()
        start = int(1_000_000)
        df.loc[mask, "Ticket"] = np.arange(start, start + mask.sum())

    # Group rows by Ticket (handles both single-row-per-trade and multi-row open/close exports)
    grouped = df.sort_values("Time").groupby("Ticket", as_index=False).agg(
        Symbol = ("Symbol", lambda s: s.dropna().astype(str).iloc[0] if len(s.dropna())>0 else "UNKNOWN"),
        OpenTime = ("Time", "first"),
        CloseTime = ("Time", "last"),
        ActionOpen = ("Action", "first"),
        Lots = ("Lots", "first"),
        EntryPrice = ("Price", "first"),
        ExitPrice = ("ExitPrice", "last"),
        Swap = ("Swap", "sum"),
        Profit = ("Profit", "sum")
    )

    # If ExitPrice missing, try to use last Price per ticket
    if grouped["ExitPrice"].isnull().all():
        last_price = df.sort_values(["Ticket","Time"]).groupby("Ticket", as_index=True).tail(1)[["Ticket","Price"]].set_index("Ticket")
        grouped = grouped.set_index("Ticket")
        if "Price" in last_price.columns:
            grouped["ExitPrice"] = grouped["ExitPrice"].fillna(last_price["Price"])
        grouped = grouped.reset_index()

    # Fill OpenTime / CloseTime safely
    grouped["OpenTime"].fillna(pd.Timestamp.now(), inplace=True)
    grouped["CloseTime"].fillna(grouped["OpenTime"], inplace=True)

    # Fallback Profit calculation where Profit is missing or zero
    def compute_fallback_profit(row):
        p = row.get("Profit")
        if pd.notna(p) and not math.isnan(p) and float(p) != 0.0:
            return float(p)
        ep = row.get("EntryPrice")
        xp = row.get("ExitPrice")
        lots = row.get("Lots") if not pd.isna(row.get("Lots")) else 0.0
        try:
            if pd.notna(ep) and pd.notna(xp) and pd.notna(lots):
                return float((xp - ep) * lots)
        except Exception:
            pass
        return 0.0

    grouped["Profit"] = grouped.apply(compute_fallback_profit, axis=1)

    # Derived analytics fields
    grouped["Date"] = pd.to_datetime(grouped["OpenTime"], errors="coerce").dt.normalize()
    grouped["Win"] = grouped["Profit"] > 0
    grouped["Loss"] = grouped["Profit"] < 0
    grouped["BreakEven"] = grouped["Profit"] == 0
    grouped["CumProfit"] = grouped["Profit"].cumsum()

    if "ActionOpen" in grouped.columns:
        grouped["ActionOpen"] = grouped["ActionOpen"].astype(str).str.upper().replace({"BUY":"BUY","SELL":"SELL"})

    return df, grouped.reset_index(drop=True)

def make_demo_data(n=240):
    rng = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='12h')
    syms = ['USDINDEX','SPX500','US2000','MICROSOFT','META','TESLA','AMD','XBI','WTI']
    rows=[]
    for i, dt in enumerate(rng[::-1]):
        p = round(np.random.normal(0, 250),2)
        rows.append({
            'Ticket': 10000+i,
            'Symbol': np.random.choice(syms),
            'Time': dt.strftime("%Y-%m-%d %H:%M:%S"),
            'Action': np.random.choice(['BUY','SELL']),
            'Lots': np.random.choice([0.01,0.05,0.1,0.2]),
            'Price': np.random.uniform(10,500),
            'Swap': np.random.uniform(-1,1),
            'Profit': p
        })
    return pd.DataFrame(rows)

def profit_factor(df):
    if df is None or len(df)==0: return 0.0
    gp = df.loc[df["Profit"]>0,"Profit"].sum()
    gl = -df.loc[df["Profit"]<0,"Profit"].sum()
    if gl==0 and gp>0: return np.inf
    if gl==0: return 0.0
    return gp/gl

def trade_expectancy(df):
    if df is None or len(df)==0: return 0.0,0.0,0.0
    wins = df[df["Profit"]>0]["Profit"]
    losses = -df[df["Profit"]<0]["Profit"]
    avg_win = wins.mean() if len(wins)>0 else 0.0
    avg_loss = losses.mean() if len(losses)>0 else 0.0
    win_rate = len(wins)/len(df) if len(df)>0 else 0.0
    expectancy = (avg_win*win_rate - avg_loss*(1-win_rate))
    return avg_win, avg_loss, expectancy

def get_color_for_pnl(pnl):
    if pnl>0: return "#26a269"
    if pnl<0: return "#ff5b5b"
    return "#9ca3af"

def ring_svg_winpct(pct, size=48, thickness=7):
    pf = max(0.0, min(1.0, (pct or 0)/100.0))
    radius = (size-thickness)/2
    circumference = 2*math.pi*radius
    dash = circumference*pf
    stroke="#26a269"
    svg=f'''
    <svg class="ring-sz" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
      <g transform="translate({size/2},{size/2})">
        <circle r="{radius}" fill="transparent" stroke="#081226" stroke-width="{thickness}" />
        <circle r="{radius}" fill="transparent" stroke="{stroke}" stroke-width="{thickness}" stroke-dasharray="{dash} {circumference - dash}" stroke-linecap="round" transform="rotate(-90)" />
      </g>
    </svg>
    '''
    return svg

def ring_svg_pf(value, min_v=0, max_v=3, size=48, thickness=6, positive_threshold=1.0):
    radius = (size-thickness)/2
    circumference = 2*math.pi*radius
    val = max(min_v, min(max_v, value)) if value is not None else 0.0
    pct = (val-min_v)/(max_v-min_v) if max_v>min_v else 0.0
    dash = circumference*pct
    stroke="#26a269" if (value is not None and value>=positive_threshold) else "#ff5b5b"
    if value is None: stroke="#94a3b8"
    svg=f'''
    <svg class="ring-sz" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
      <g transform="translate({size/2},{size/2})">
        <circle r="{radius}" fill="transparent" stroke="#081226" stroke-width="{thickness}" />
        <circle r="{radius}" fill="transparent" stroke="{stroke}" stroke-width="{thickness}" stroke-dasharray="{dash} {circumference - dash}" stroke-linecap="round" transform="rotate(-90)" />
      </g>
    </svg>
    '''
    return svg

# ---------------- Session initialization ----------------
for key in ["imports","last_added","show_top_uploader","edit_widgets_open","last_import_ts",
            "visible_metrics","sidebar_manual_open","sidebar_auto_open","mt5_accounts","last_selected_accounts"]:
    if key not in st.session_state:
        if key=="imports":
            st.session_state[key] = {}
        elif key=="visible_metrics":
            st.session_state[key] = {"Net PnL": True, "Trade Expectancy": True, "Profit Factor": True, "Trade Win %": True, "Avg win/loss trade": True}
        elif key in ("sidebar_manual_open","sidebar_auto_open","show_top_uploader","edit_widgets_open"):
            st.session_state[key] = False
        elif key=="last_import_ts":
            st.session_state[key] = None
        elif key=="mt5_accounts":
            st.session_state[key] = []
        elif key=="last_selected_accounts":
            st.session_state[key] = []

# Ensure demo account exists
if not st.session_state["imports"]:
    raw_demo = make_demo_data(240)
    raw_demo, grouped_demo = process_mt5_df(raw_demo)
    st.session_state["imports"]["Demo Account"] = {"raw": raw_demo, "grouped": grouped_demo}
    st.session_state["last_added"] = "Demo Account"
    st.session_state["last_import_ts"] = datetime.utcnow()

# ---------------- Sidebar: Imports ----------------
st.sidebar.markdown("<div style='margin-bottom:8px;font-weight:700;color:#e6eef8'>Imports</div>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("<div class='side-buttons'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])

    # Manual import toggle
    if col1.button("📁", key="icon_manual"):
        st.session_state["sidebar_manual_open"] = not st.session_state.get("sidebar_manual_open", False)
    col1.markdown("<div class='side-label'>Manual Imports</div>", unsafe_allow_html=True)

    # Automatic MT5 import toggle
    if col2.button("🔁", key="icon_auto"):
        st.session_state["sidebar_auto_open"] = not st.session_state.get("sidebar_auto_open", False)
    col2.markdown("<div class='side-label'>Automatic Sync (MT5)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Manual Imports ----------------
if st.session_state.get("sidebar_manual_open", False):
    st.sidebar.markdown("<div class='small-muted-2'>Manual — upload MT5 CSV/XLSX or other broker CSVs</div>", unsafe_allow_html=True)
    uploaded_files = st.sidebar.file_uploader("Upload file(s)", accept_multiple_files=True, key="sidebar_upload")
    account_name = st.sidebar.text_input("Account name (optional)", key="sidebar_account_name")
    if st.sidebar.button("Add upload(s)"):
        if not uploaded_files:
            st.sidebar.warning("Choose file(s) first")
        else:
            if "imports" not in st.session_state:
                st.session_state["imports"] = {}
            for f in uploaded_files:
                try:
                    # read raw file
                    if f.name.lower().endswith((".csv", ".txt")):
                        raw_df = pd.read_csv(f)
                    else:
                        raw_df = pd.read_excel(f, engine="openpyxl")

                    # Universal normalization: use process_mt5_df which is robust to many formats
                    raw_norm, grouped = process_mt5_df(raw_df)
                    key = account_name.strip() or f.name
                    base, i = key, 1
                    while key in st.session_state["imports"]:
                        key = f"{base} ({i})"; i += 1
                    st.session_state["imports"][key] = {"raw": raw_norm, "grouped": grouped}
                    st.session_state["last_added"] = key
                    st.session_state["last_import_ts"] = datetime.utcnow()
                    st.sidebar.success(f"Added {key}")
                except Exception as e:
                    st.sidebar.error(f"Failed {f.name}: {e}")

# ---------------- Automatic MT5 Imports ----------------
if st.session_state.get("sidebar_auto_open", False):
    st.sidebar.markdown("<div class='small-muted-2'>Automatic — connect to MT5</div>", unsafe_allow_html=True)
    login = st.text_input("Login", key="mt5_login")
    password = st.text_input("Password", type="password", key="mt5_password")
    server = st.text_input("Server", key="mt5_server")
    acct_label = st.text_input("Account label (optional)", key="mt5_label")

    if "mt5_accounts" not in st.session_state:
        st.session_state["mt5_accounts"] = []

    from datetime import datetime as _dt, timedelta as _td

    def connect_and_fetch_mt5(login, password, server, label=None):
        """Connects to MT5, fetches trade history, stores in session_state."""
        try:
            if not mt5.initialize():
                return False, f"MT5 initialize() failed: {mt5.last_error()}"
            if not mt5.login(int(login), password=password, server=server):
                mt5.shutdown()
                return False, f"MT5 login failed: {mt5.last_error()}"
            end_time = _dt.utcnow()
            start_time = end_time - _td(days=365)
            trades = mt5.history_deals_get(start_time, end_time)
            if not trades:
                df = pd.DataFrame()
                grouped = pd.DataFrame()
            else:
                df = pd.DataFrame([t._asdict() for t in trades])
                df, grouped = process_mt5_df(df)

            key = label.strip() if label else f"MT5-{login}"
            base, i = key, 1
            if "imports" not in st.session_state:
                st.session_state["imports"] = {}
            while key in st.session_state["imports"]:
                key = f"{base} ({i})"
                i += 1
            st.session_state["imports"][key] = {"raw": df, "grouped": grouped}
            st.session_state["last_added"] = key
            st.session_state["last_import_ts"] = datetime.utcnow()
            if not any(acct.get("login")==login and acct.get("server")==server for acct in st.session_state["mt5_accounts"]):
                st.session_state["mt5_accounts"].append({"login": login, "server": server, "label": key})
            mt5.shutdown()
            return True, f"Imported MT5 account {key}"
        except Exception as e:
            try:
                mt5.shutdown()
            except Exception:
                pass
            return False, str(e)

    if st.sidebar.button("Connect & Fetch MT5"):
        if not (login and password and server):
            st.sidebar.error("Fill all MT5 fields")
        else:
            success, msg = connect_and_fetch_mt5(login, password, server, acct_label)
            if success:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

    # list previously connected accounts
    if st.session_state["mt5_accounts"]:
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        for i, acct in enumerate(st.session_state["mt5_accounts"]):
            cols = st.sidebar.columns([3,1])
            cols[0].markdown(f"**{acct['label']}**")
            if cols[1].button("Sync now", key=f"sync_{i}"):
                success, msg = connect_and_fetch_mt5(acct["login"], password="", server=acct["server"], label=acct["label"])
                if success:
                    st.sidebar.success(f"Synced {acct['label']}")
                else:
                    st.sidebar.error(f"Failed syncing {acct['label']}: {msg}")

# ---------------- Prepare filtered trades safely ----------------
if "imports" not in st.session_state:
    st.session_state["imports"] = {}

import_names = list(st.session_state["imports"].keys())
if not import_names:
    st.sidebar.info("No imports yet")
    raw_df = pd.DataFrame()
    trades = pd.DataFrame(columns=["Ticket","Symbol","OpenTime","CloseTime","Profit","Date","ActionOpen"])
    filtered = trades.copy()
else:
    if not st.session_state.get("last_selected_accounts"):
        st.session_state["last_selected_accounts"] = [st.session_state.get("last_added", import_names[0])]
    selected_accounts = st.sidebar.multiselect("Account(s)", import_names, default=st.session_state["last_selected_accounts"])
    st.session_state["last_selected_accounts"] = selected_accounts

    combined_raw, combined_grouped = [], []
    for name in selected_accounts:
        entry = st.session_state["imports"].get(name, {"raw": pd.DataFrame(), "grouped": pd.DataFrame()})
        df_raw = entry["raw"].copy() if entry["raw"] is not None else pd.DataFrame()
        df_group = entry["grouped"].copy() if entry["grouped"] is not None else pd.DataFrame()
        if not df_raw.empty:
            df_raw["Account"] = name
        if not df_group.empty:
            df_group["Account"] = name
        combined_raw.append(df_raw)
        combined_grouped.append(df_group)

    raw_df = pd.concat([d for d in combined_raw if not d.empty], ignore_index=True) if combined_raw else pd.DataFrame()
    trades = pd.concat([g for g in combined_grouped if not g.empty], ignore_index=True) if combined_grouped else pd.DataFrame()

    # Ensure 'Date' exists
    if trades.empty:
        trades = pd.DataFrame(columns=["Ticket","Symbol","OpenTime","CloseTime","Profit","Date","ActionOpen"])
    if "Date" not in trades.columns and "OpenTime" in trades.columns:
        trades["Date"] = pd.to_datetime(trades["OpenTime"], errors="coerce").dt.normalize()
    else:
        trades["Date"] = pd.to_datetime(trades.get("Date"), errors="coerce")

    # Filter trades safely
    if not trades.empty:
        symbols_all = sorted(trades["Symbol"].fillna("UNKNOWN").unique())
        sel_syms = st.sidebar.multiselect("Symbols", symbols_all, default=symbols_all)
        date_min = trades["Date"].min().date()
        date_max = trades["Date"].max().date()
        date_range = st.sidebar.date_input("Date range", value=(date_min, date_max))
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(hours=23, minutes=59)
        filtered = trades[trades["Symbol"].isin(sel_syms) & trades["Date"].between(start_dt, end_dt)].copy()
    else:
        filtered = pd.DataFrame()

# ---------------- Compute metrics ----------------
total_trades = len(filtered)
total_profit = filtered["Profit"].sum() if "Profit" in filtered.columns else 0.0
pf = profit_factor(filtered)
avg_win, avg_loss, expectancy = trade_expectancy(filtered)
wins = filtered["Win"].sum() if "Win" in filtered.columns else filtered[filtered.get("Profit",0)>0].shape[0]
win_rate = (wins / total_trades * 100) if total_trades else 0.0
avg_win_loss_ratio = (avg_win/avg_loss) if avg_loss>0 else (avg_win if avg_win>0 else 0.0)

# ---------------- Header (dynamic greeting using EAT Nairobi time + user name) ----------------
from datetime import datetime, timedelta

# Nairobi is UTC+3 (EAT)
now_utc = datetime.utcnow()
now_eat = now_utc + timedelta(hours=3)
hour = now_eat.hour

# Determine greeting
if 5 <= hour < 12:
    greet = "Good morning"
elif 12 <= hour < 17:
    greet = "Good afternoon"
else:
    greet = "Good evening"

# ---------------- Fetch Name From Profiles ----------------
user_name = None

if st.session_state.get("user"):
    user_id = st.session_state.user.id

    try:
        profile_res = supabase.table("profiles") \
            .select("full_name") \
            .eq("id", user_id) \
            .single() \
            .execute()

        if profile_res.data:
            user_name = profile_res.data["full_name"]

    except Exception:
        user_name = None

# Combine greeting + name
if user_name:
    greet_text = f"{greet}, {user_name}!"
else:
    greet_text = f"{greet}!"

# ---------------- Layout ----------------
header_left, header_right = st.columns([1, 2])

with header_left:
    st.markdown(
        f"<div style='padding:6px 0; font-size:16px; font-weight:700'>{greet_text}</div>",
        unsafe_allow_html=True
    )

with header_right:
    cols = st.columns([2, 1, 1, 1])

    last_ts = st.session_state.get("last_import_ts")

    if last_ts:
        last_import_eat = last_ts + timedelta(hours=3)
        last_import_text = last_import_eat.strftime("%Y-%m-%d %H:%M:%S")
    else:
        last_import_text = "No imports yet"

    cols[1].markdown(
        f"<div style='text-align:right; color:#9CA3AF; font-size:13px'>Last import: "
        f"<strong style='color:white'>{last_import_text}</strong></div>",
        unsafe_allow_html=True
    )

    if cols[2].button("Edit Widgets"):
        st.session_state["edit_widgets_open"] = not st.session_state.get("edit_widgets_open", False)

    if cols[3].button("+ Import trades"):
        st.session_state["show_top_uploader"] = not st.session_state.get("show_top_uploader", False)
# ---------------- Inline uploader ----------------
if st.session_state.get("show_top_uploader", False):
    st.markdown("<div class='card' style='margin-top:8px'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;align-items:center;gap:12px'>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop files here or click to browse (CSV / XLSX)",
        accept_multiple_files=True,
        key="top_uploader"
    )
    top_name = st.text_input("Name this import (optional)", key="top_name")

    if st.button("Add upload(s) (top)"):
        if not uploaded:
            st.warning("Choose file(s) first")
        else:
            added = []
            for f in uploaded:
                try:
                    if f.name.lower().endswith((".csv", ".txt")):
                        raw = pd.read_csv(f)
                    else:
                        raw = pd.read_excel(f, engine="openpyxl")

                    raw_norm, grouped = process_mt5_df(raw)

                    key = top_name.strip() or f.name
                    base, i = key, 1
                    while key in st.session_state["imports"]:
                        key = f"{base} ({i})"
                        i += 1

                    st.session_state["imports"][key] = {"raw": raw_norm, "grouped": grouped}
                    added.append(key)
                except Exception as e:
                    st.error(f"Failed {f.name}: {e}")

            if added:
                st.success(f"Added: {', '.join(added)}")
                st.session_state["last_added"] = added[-1]
                st.session_state["last_import_ts"] = datetime.utcnow()  # Store UTC

    st.markdown("</div>", unsafe_allow_html=True)

# Edit widgets toggles
if st.session_state["edit_widgets_open"]:
    with st.expander("Edit Widgets — Toggle top metrics visibility", expanded=True):
        v = st.session_state["visible_metrics"]
        new_v = {}
        for k in v:
            new_v[k] = st.checkbox(k, value=v[k], key=f"vis_{k}")
        st.session_state["visible_metrics"] = new_v
# ---------------- Top metrics ----------------
col_net, col_exp, col_pf, col_winpct, col_awl = st.columns([1.2, 0.8, 0.8, 0.8, 1.2])

with col_net:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Net PnL</div>", unsafe_allow_html=True)
    pnl_color = "#26a269" if total_profit > 0 else ("#ff5b5b" if total_profit < 0 else "#9ca3af")
    pnl_text = f"${total_profit:,.2f}"
    st.markdown(f"<div class='kpi-val'><div style='font-size:22px;font-weight:800;color:{pnl_color}'>{pnl_text}</div><div style='font-size:12px;color:#94a3b8'>trades {total_trades}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_exp:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Trade Expectancy</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpi-val'><div style='font-size:18px;font-weight:700'>${expectancy:.2f}</div><div style='font-size:12px;color:#94a3b8'>avg ${avg_win:.2f} / ${avg_loss:.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pf:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Profit Factor</div>", unsafe_allow_html=True)
    pf_display = pf if (not math.isinf(pf) and not math.isnan(pf)) else (pf if math.isinf(pf) else 0.0)
    svg_pf = ring_svg_pf(pf_display, min_v=0, max_v=3, size=48, thickness=6, positive_threshold=1.0)
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div style='font-weight:800;font-size:18px'>{(f'{pf_display:.2f}' if (pf_display is not None and not math.isinf(pf_display)) else '∞')}</div>{svg_pf}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_winpct:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Trade Win %</div>", unsafe_allow_html=True)
    svg_win = ring_svg_winpct(win_rate, size=48, thickness=7)
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div style='font-weight:800;font-size:18px'>{win_rate:.2f}%</div>{svg_win}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_awl:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Avg win/loss trade</div>", unsafe_allow_html=True)

    try:
        avg_win_val = round(filtered[filtered["Profit"] > 0]["Profit"].mean() if not filtered[filtered["Profit"] > 0].empty else 0, 2)
    except Exception:
        avg_win_val = 0.0
    try:
        avg_loss_val = round(abs(filtered[filtered["Profit"] < 0]["Profit"].mean()) if not filtered[filtered["Profit"] < 0].empty else 0, 2)
    except Exception:
        avg_loss_val = 0.0

    total = avg_win_val + avg_loss_val
    if total > 0:
        left_pct = int(round((avg_win_val / total) * 100))
    else:
        left_pct = 50
    right_pct = 100 - left_pct

    bar_html = (
        f"<div style='width:100%; margin-top:6px;'>"
        f"  <div style='height:10px; width:100%; display:flex; border-radius:8px; overflow:hidden;'>"
        f"    <div style='width:{left_pct}%; background:#26a269;'></div>"
        f"    <div style='width:{right_pct}%; background:#ff5b5b;'></div>"
        f"  </div>"
        f"  <div style='display:flex; justify-content:space-between; margin-top:6px; font-size:13px; font-weight:700;'>"
        f"    <div style='color:#26a269;'>${avg_win_val:.2f}</div>"
        f"    <div style='color:#ff5b5b;'>${avg_loss_val:.2f}</div>"
        f"  </div>"
        f"</div>"
    )

    st.markdown(bar_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Mid area ----------------
col_left, col_mid, col_right = st.columns([1.4, 2.4, 1.4])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700'>Zella Score <span style='color:#94a3b8;font-weight:400;font-size:12px'>Beta</span></div>", unsafe_allow_html=True)

    z_win = win_rate / 100.0
    z_pf = 0.0 if math.isinf(pf) else min((pf / 3.0) if pf != 0 else 0.0, 1.0)
    z_awl = min((avg_win_loss_ratio / 3.0) if avg_win_loss_ratio > 0 else 0.0, 1.0)
    z_score = 100 * (0.4 * z_win + 0.3 * z_awl + 0.3 * z_pf)

    st.markdown(
        f"<div style='margin-top:8px;font-size:20px;font-weight:800;color:#26a269'>Your Zella Score: {z_score:.1f}</div>",
        unsafe_allow_html=True
    )

    try:
        categories = ["Win %", "Avg win/loss", "Profit factor"]
        vals = [z_win * 100, z_awl * 100, z_pf * 100]
        fig_r = go.Figure()
        fig_r.add_trace(
            go.Scatterpolar(
                theta=categories + [categories[0]],
                r=vals + [vals[0]],
                fill="toself",
                line_color="#9CA3AF"
            )
        )
        fig_r.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False)),
            height=260,
            margin=dict(t=10, b=4, l=4, r=4),
        )
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        st.info("Unable to render Zella triangle.")

    st.markdown("</div>", unsafe_allow_html=True)


with col_mid:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Daily Net Cumulative P&L</strong>", unsafe_allow_html=True)
    try:
        if len(filtered) > 0:
            daily = (
                filtered
                .groupby(filtered["Date"].dt.date)
                .agg(DailyPnL=("Profit", "sum"))
                .sort_index()
            )
            daily["Cumulative"] = daily["DailyPnL"].cumsum()

            fig_area = go.Figure()

            # ----- Positive cumulative (fade to zero) -----
            y_pos = [val if val > 0 else 0 for val in daily["Cumulative"]]
            fig_area.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=y_pos,
                    mode="lines",
                    line=dict(color="#26a269", width=2),
                    fill="tozeroy",
                    fillgradient=dict(
                        type="vertical",
                        colorscale=[
                            [0.0, "rgba(38,162,105,0.0)"],
                            [1.0, "rgba(38,162,105,0.45)"]
                        ]
                    ),
                    showlegend=False
                )
            )

            # ----- Negative cumulative (fade to zero) -----
            y_neg = [val if val < 0 else 0 for val in daily["Cumulative"]]
            fig_area.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=y_neg,
                    mode="lines",
                    line=dict(color="#ff4c4c", width=2),
                    fill="tozeroy",
                    fillgradient=dict(
                        type="vertical",
                        colorscale=[
                            [0.0, "rgba(255,76,76,0.45)"],
                            [1.0, "rgba(255,76,76,0.0)"]
                        ]
                    ),
                    showlegend=False
                )
            )

            # ----- Zero line -----
            fig_area.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=[0] * len(daily),
                    mode="lines",
                    line=dict(color="#0f1724", width=1),
                    showlegend=False
                )
            )

            fig_area.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=False, color="#94a3b8"),
                yaxis=dict(showgrid=False, color="#94a3b8"),
            )

            st.plotly_chart(fig_area, use_container_width=True)

        else:
            st.info("No trades in range for cumulative P&L.")
    except Exception as e:
        st.error(f"Failed to draw cumulative P&L: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    
with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Net Daily P&L</strong>", unsafe_allow_html=True)

    try:
        if len(filtered) > 0:
            daily_local = (
                filtered
                .groupby(filtered["Date"].dt.date)
                .agg(DailyPnL=("Profit", "sum"))
                .sort_index()
            )

            colors = [get_color_for_pnl(x) for x in daily_local["DailyPnL"]]

            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=daily_local.index,
                    y=daily_local["DailyPnL"],
                    marker_color=colors
                )
            )

            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=False, color="#94a3b8"),
                yaxis=dict(showgrid=False, color="#94a3b8"),
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No daily P&L data.")
    except Exception as e:
        st.error(f"Failed to draw daily P&L: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Bottom row: Recent Trades and Calendar ----------------
left_bot, right_bot = st.columns([1.6, 1.4])

with left_bot:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<strong>Recent Trades</strong>", unsafe_allow_html=True)
    try:
        recent = filtered.sort_values("OpenTime", ascending=False).head(12)[["CloseTime","Symbol","Profit"]].copy()
        if len(recent)>0:
            recent["CloseTime"] = pd.to_datetime(recent["CloseTime"], errors="coerce").dt.strftime("%m/%d/%Y")
            recent["ProfitFmt"] = recent["Profit"].apply(lambda x: f"${x:,.2f}")
            html = "<table class='smalltbl'><thead><tr><th>Close Date</th><th>Symbol</th><th>Net P&L</th></tr></thead><tbody>"
            for _, r in recent.iterrows():
                try:
                    pv = float(r["Profit"])
                except Exception:
                    pv = 0.0
                color = "color:#26a269" if pv>0 else ("color:#ff5b5b" if pv<0 else "color:#9ca3af")
                html += f"<tr><td>{r['CloseTime']}</td><td>{r['Symbol']}</td><td style='{color}'>{r['ProfitFmt']}</td></tr>"
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No recent trades to show.")
    except Exception as e:
        st.error(f"Failed to render recent trades: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with right_bot:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex;justify-content:space-between;align-items:center'>"
        "<strong>Calendar / Activity</strong></div>",
        unsafe_allow_html=True
    )

    # ---------------- Calendar month state (EAT) ----------------
    if "calendar_month" not in st.session_state:
        today = (datetime.utcnow() + timedelta(hours=3)).date()  # Nairobi time
        st.session_state["calendar_month"] = date(today.year, today.month, 1)

    # ---------------- Header controls ----------------
    cal_col_left, cal_col_title, cal_col_right = st.columns([1, 2, 1])

    with cal_col_left:
        if st.button("◀"):
            cur = st.session_state["calendar_month"]
            st.session_state["calendar_month"] = (
                (cur.replace(day=1) - timedelta(days=1)).replace(day=1)
            )

    display_start = st.session_state["calendar_month"].replace(day=1)
    display_end = (pd.Timestamp(display_start) + pd.offsets.MonthEnd(0)).date()

    cal_col_title.markdown(
        f"<div style='text-align:center;font-weight:700;font-size:14px'>"
        f"{display_start.strftime('%B %Y')}</div>",
        unsafe_allow_html=True
    )

    with cal_col_right:
        if st.button("▶"):
            cur = st.session_state["calendar_month"]
            st.session_state["calendar_month"] = (
                (cur.replace(day=28) + timedelta(days=8)).replace(day=1)
            )

    # ---------------- Build stats map ----------------
    if len(filtered) > 0:
        stats = (
            filtered
            .groupby(filtered["Date"].dt.date)
            .agg(DailyPnL=("Profit", "sum"), Trades=("Ticket", "count"))
            .reset_index()
        )
        stats_map = {
            r["Date"]: {"pnl": r["DailyPnL"], "trades": int(r["Trades"])}
            for _, r in stats.iterrows()
        }
    else:
        stats_map = {}

    # ---------------- Monthly total ----------------
    month_mask = (
        (filtered["Date"].dt.date >= display_start)
        & (filtered["Date"].dt.date <= display_end)
        if len(filtered) > 0 else pd.Series([], dtype=bool)
    )

    monthly_total = filtered.loc[month_mask, "Profit"].sum() if len(filtered) > 0 else 0.0
    monthly_color = "#26a269" if monthly_total > 0 else (
        "#ff5b5b" if monthly_total < 0 else "#9ca3af"
    )

    st.markdown(
        f"<div style='text-align:right;font-weight:700;font-size:13px;color:{monthly_color}'>"
        f"Monthly total PnL: ${monthly_total:+,.2f}</div>",
        unsafe_allow_html=True
    )

    # ---------------- Calendar range ----------------
    first_sunday = display_start - timedelta(days=(display_start.weekday() + 1) % 7)
    last_saturday = display_end + timedelta(days=(6 - display_end.weekday()) % 7)
    all_dates = pd.date_range(first_sunday, last_saturday).date

    # ---------------- Calendar HTML ----------------
    cal_html = "<div class='calendar-grid'>"

    # Day headers
    day_headers = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "W"]
    for d in day_headers:
        cal_html += f"<div class='calendar-header'>{d}</div>"

    weekly_total = 0
    weekly_trades = 0

    for dt in all_dates:
        data = stats_map.get(dt, {"pnl": 0, "trades": 0})

        if display_start <= dt <= display_end:
            cls = (
                "positive" if data["pnl"] > 0
                else "negative" if data["pnl"] < 0
                else "neutral"
            )
            cal_html += (
                f"<div class='calendar-day {cls}'>"
                f"<div class='day-num'>{dt.day}</div>"
                f"<div style='font-size:11px;text-align:center'>"
                f"{data['trades']} trades<br>${data['pnl']:+,.0f}</div></div>"
            )
        else:
            cal_html += (
                f"<div class='calendar-day neutral' style='opacity:0.25'>"
                f"<div class='day-num'>{dt.day}</div></div>"
            )

        weekly_total += data["pnl"]
        weekly_trades += data["trades"]

        # Saturday → weekly summary column
        if dt.weekday() == 5:
            cls_week = (
                "positive" if weekly_total > 0
                else "negative" if weekly_total < 0
                else "neutral"
            )
            week_label = "W" if weekly_total >= 0 else "L"
            week_color = "#26a269" if weekly_total >= 0 else "#ff5b5b"

            cal_html += (
                f"<div class='calendar-day weekly-summary' "
                f"style='background:#111827;color:{week_color}'>"
                f"<div class='day-num'>{week_label}</div>"
                f"<div style='font-size:12px'>{weekly_trades} trades<br>"
                f"${weekly_total:+,.0f}</div></div>"
            )

            weekly_total = 0
            weekly_trades = 0

    cal_html += "</div>"

    st.markdown(cal_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Asset Performance & Insights (robust) ----------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(
    "<div style='display:flex;justify-content:space-between;align-items:center'>"
    "<strong>Asset Performance & Trade Insights</strong>"
    "<div class='small-muted'>Pie, long/short, duration + recommendations</div></div>",
    unsafe_allow_html=True
)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if len(filtered) > 0:
    try:
        # Normalize ActionOpen column
        if "ActionOpen" not in filtered.columns:
            filtered["ActionOpen"] = filtered.get("Action", "").astype(str).str.upper()

        # Strip spaces from all columns
        filtered.columns = filtered.columns.str.strip()

        # ---------------- Left: Trades per Asset ----------------
        col_a, col_b, col_c = st.columns([1.2, 1, 1])
        with col_a:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Trades per Asset</div>", unsafe_allow_html=True)
            cnts = filtered.groupby("Symbol")["Ticket"].nunique().sort_values(ascending=False)
            if len(cnts) == 0:
                st.info("No asset trades to show.")
            else:
                labels = cnts.index.tolist()
                values = cnts.values.tolist()
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, sort=False)])
                fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)

        # ---------------- Middle: Long vs Short ----------------
        with col_b:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Long vs Short</div>", unsafe_allow_html=True)
            actions = filtered["ActionOpen"].fillna("UNKNOWN").str.upper()
            long_mask = actions.str.contains("BUY")
            short_mask = actions.str.contains("SELL")
            long_cnt = int(long_mask.sum())
            short_cnt = int(short_mask.sum())
            avg_long = filtered.loc[long_mask, "Profit"].mean() if long_cnt > 0 else 0.0
            avg_short = filtered.loc[short_mask, "Profit"].mean() if short_cnt > 0 else 0.0
            better_side = "Long" if avg_long > avg_short else ("Short" if avg_short > avg_long else "Tie")

            st.markdown(
                f"<div class='insight-card'>"
                f"<div style='font-weight:700'>Counts</div>"
                f"<div style='display:flex;justify-content:space-between;margin-top:6px'>"
                f"<div>Long: <strong>{long_cnt}</strong></div>"
                f"<div>Short: <strong>{short_cnt}</strong></div></div>"
                f"<div style='height:8px'></div>"
                f"<div style='font-weight:700'>Average P&L</div>"
                f"<div style='display:flex;justify-content:space-between;margin-top:6px'>"
                f"<div style='color:#26a269'>Long: <strong>${avg_long:,.2f}</strong></div>"
                f"<div style='color:#ff5b5b'>Short: <strong>${avg_short:,.2f}</strong></div></div>"
                f"<div style='margin-top:8px;font-weight:700'>Better: <span style='color:#fff'>{better_side}</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # ---------------- Right: Average Trade Duration ----------------
        with col_c:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Average Trade Duration</div>", unsafe_allow_html=True)
            try:
                import re
                durations = []

                # Detect relevant columns
                open_cols = ["OpenTime", "Open Date", "Open"]
                close_cols = ["CloseTime", "Close Date", "Close"]
                single_cols = ["Time", "Timestamp"]

                open_col = next((c for c in open_cols if c in filtered.columns), None)
                close_col = next((c for c in close_cols if c in filtered.columns), None)
                single_col = next((c for c in single_cols if c in filtered.columns), None)

                for _, row in filtered.iterrows():
                    # Case 1: Duration column exists
                    if "Duration" in filtered.columns and isinstance(row.get("Duration"), str) and row["Duration"].strip() != "":
                        dur_str = row["Duration"]
                        hours = int(re.search(r'(\d+)\s*h', dur_str).group(1)) if re.search(r'(\d+)\s*h', dur_str) else 0
                        minutes = int(re.search(r'(\d+)\s*m', dur_str).group(1)) if re.search(r'(\d+)\s*m', dur_str) else 0
                        seconds = int(re.search(r'(\d+)\s*s', dur_str).group(1)) if re.search(r'(\d+)\s*s', dur_str) else 0
                        durations.append(hours + minutes/60 + seconds/3600)

                    # Case 2: Open/Close columns exist → compute duration
                    elif open_col and close_col:
                        try:
                            ot = pd.to_datetime(row[open_col], errors="coerce")
                            ct = pd.to_datetime(row[close_col], errors="coerce")
                            if pd.notnull(ot) and pd.notnull(ct):
                                delta_hours = (ct - ot).total_seconds() / 3600
                                durations.append(delta_hours)
                        except Exception:
                            continue

                    # Case 3: Single timestamp only → skip or append 0
                    elif single_col:
                        durations.append(np.nan)

                # Convert to Series and remove invalid values
                durations = pd.Series(durations).replace([np.inf, -np.inf], np.nan).dropna()

                avg_dur = durations.mean() if len(durations) > 0 else 0.0
                med_dur = durations.median() if len(durations) > 0 else 0.0

                st.markdown(
                    f"<div class='insight-card'>"
                    f"<div>Average: <strong>{avg_dur:.2f} hrs</strong></div>"
                    f"<div>Median: <strong>{med_dur:.2f} hrs</strong></div>"
                    f"<div style='height:8px'></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                # Histogram
                if len(durations) > 0:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Histogram(x=durations, nbinsx=20))
                    fig_h.update_layout(
                        height=160,
                        margin=dict(t=10, b=10, l=10, r=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                else:
                    st.info("No duration data to display.")

            except Exception as e:
                st.info(f"Duration data unavailable: {e}")

    except Exception as e:
        st.error(f"Failed computing asset insights: {e}")

    # ---------------- Bottom row of recommendations: assets to focus, most traded session, best session ----------------
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1,1,1])
    # derive sessions in EAT (UTC+3) and classify per trade by OpenTime
    try:
        df_sessions = filtered.copy()
        df_sessions["OpenTime_dt"] = pd.to_datetime(df_sessions["OpenTime"], errors="coerce")
        # shift to EAT (assume times are UTC naive -> interpret as UTC then add 3 hours)
        df_sessions["OpenTime_eat"] = df_sessions["OpenTime_dt"] + pd.Timedelta(hours=3)
        df_sessions["hour_eat"] = df_sessions["OpenTime_eat"].dt.hour.fillna(-1).astype(int)

        def session_from_hour(h):
            # Define sessions in EAT:
            # Asian: 00:00 - 07:59
            # London: 08:00 - 15:59
            # NY: 16:00 - 23:59
            if h < 0:
                return "Unknown"
            if 0 <= h < 8:
                return "Asian"
            if 8 <= h < 16:
                return "London"
            return "NY"

        df_sessions["Session"] = df_sessions["hour_eat"].apply(session_from_hour)
        session_counts = df_sessions.groupby("Session")["Ticket"].nunique().sort_values(ascending=False)
        session_perf = df_sessions.groupby("Session")["Profit"].mean().sort_values(ascending=False)

        # Left: Which assets to capitalize on (top performers)
        with r1:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Assets to Focus On</div>", unsafe_allow_html=True)
            asset_perf = filtered.groupby("Symbol")["Profit"].sum().sort_values(ascending=False)
            if len(asset_perf) == 0:
                st.info("No asset performance data.")
            else:
                top_assets = asset_perf.head(3)
                for sym, val in top_assets.items():
                    color = "#26a269" if val >= 0 else "#ff5b5b"
                    st.markdown(f"<div class='insight-card'><div style='font-weight:700'>{sym}</div><div>Net P&L: <span style='color:{color}'>${val:,.2f}</span></div></div>", unsafe_allow_html=True)

        # Middle: Most traded session
        with r2:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Most Traded Session</div>", unsafe_allow_html=True)
            if len(session_counts) == 0:
                st.info("No session trades.")
            else:
                most_traded = session_counts.index[0]
                st.markdown(f"<div class='insight-card'><div style='font-weight:700'>{most_traded}</div><div style='margin-top:6px'>Trades: <strong>{int(session_counts.iloc[0])}</strong></div></div>", unsafe_allow_html=True)

        # Right: Session to focus on (best average P&L)
        with r3:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Best Performing Session</div>", unsafe_allow_html=True)
            if len(session_perf) == 0:
                st.info("No session performance.")
            else:
                best_session = session_perf.index[0]
                st.markdown(f"<div class='insight-card'><div style='font-weight:700'>{best_session}</div><div style='margin-top:6px'>Avg P&L: <strong>${session_perf.iloc[0]:,.2f}</strong></div></div>", unsafe_allow_html=True)
    except Exception as e:
        st.info("Not enough data to compute session recommendations.")

    # ---------------- NEW ROW: Other Sessions, Worst/Best Days, Worst/Best Trade Times ----------------
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    o1, o2, o3, t1, t2 = st.columns([1,1,1,1,1])  # added t1 and t2 for trade times

    try:
        # Ensure Date and Profit exist
        filtered["Date"] = pd.to_datetime(filtered["Date"], errors="coerce")
        filtered = filtered.dropna(subset=["Date", "Profit"])

        # Add weekday name
        filtered["weekday"] = filtered["Date"].dt.day_name()

        # -------------------- Other Sessions --------------------
        with o1:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Other Sessions Traded & Performance</div>", unsafe_allow_html=True)
            if 'session_counts' not in locals() or len(session_counts) == 0:
                st.info("No session data.")
            else:
                html = "<div class='insight-card'><table style='width:100%; font-size:13px; border-collapse:collapse; text-align:center;'>"
                html += "<tr><th style='color:#94a3b8; padding-bottom:6px;'>Session</th><th style='color:#94a3b8; padding-bottom:6px;'>Trades</th><th style='color:#94a3b8; padding-bottom:6px;'>Avg P&L</th></tr>"
                for s in session_counts.index:
                    trades_cnt = int(session_counts.loc[s])
                    avg_p = session_perf.loc[s] if s in session_perf.index else 0.0
                    color = "#26a269" if avg_p >= 0 else "#ff5b5b"
                    html += f"<tr><td>{s}</td><td>{trades_cnt}</td><td style='color:{color}'>${avg_p:,.2f}</td></tr>"
                html += "</table></div>"
                st.markdown(html, unsafe_allow_html=True)

        # -------------------- Worst Performing Weekdays --------------------
        with o2:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Worst Performing Days</div>", unsafe_allow_html=True)
            if filtered.empty:
                st.info("No trade data available.")
            else:
                weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                weekday_stats = filtered.groupby("weekday")["Profit"].sum().reindex(weekday_order).dropna()
                min_pnl = weekday_stats.min()
                worst_days = weekday_stats[weekday_stats == min_pnl]

                html = "<div class='insight-card'><table style='width:100%; font-size:13px; border-collapse:collapse; text-align:center;'>"
                html += "<tr><th style='color:#94a3b8; padding-bottom:6px;'>Day</th><th style='color:#94a3b8; padding-bottom:6px;'>Total P&L</th></tr>"
                for day, pnl in worst_days.items():
                    color = "#ff5b5b" if pnl < 0 else "#26a269"
                    html += f"<tr><td>{day}</td><td style='color:{color}'>${pnl:,.2f}</td></tr>"
                html += "</table></div>"
                st.markdown(html, unsafe_allow_html=True)

        # -------------------- Best Performing Weekdays --------------------
        with o3:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Best Performing Days</div>", unsafe_allow_html=True)
            if filtered.empty:
                st.info("No trade data available.")
            else:
                max_pnl = weekday_stats.max()
                best_days = weekday_stats[weekday_stats == max_pnl]

                html = "<div class='insight-card'><table style='width:100%; font-size:13px; border-collapse:collapse; text-align:center;'>"
                html += "<tr><th style='color:#94a3b8; padding-bottom:6px;'>Day</th><th style='color:#94a3b8; padding-bottom:6px;'>Total P&L</th></tr>"
                for day, pnl in best_days.items():
                    color = "#26a269" if pnl >= 0 else "#ff5b5b"
                    html += f"<tr><td>{day}</td><td style='color:{color}'>${pnl:,.2f}</td></tr>"
                html += "</table></div>"
                st.markdown(html, unsafe_allow_html=True)

        # -------------------- Worst/Best Trade Time Windows --------------------
        filtered["OpenTime"] = pd.to_datetime(filtered["OpenTime"], errors="coerce")
        filtered = filtered.dropna(subset=["OpenTime"])

        # Convert OpenTime to EAT
        filtered["OpenTime_eat"] = filtered["OpenTime"] + pd.Timedelta(hours=3)

        # Create 30-min bins
        def time_bin(dt):
            m_bin = 0 if dt.minute < 30 else 30
            start = dt.replace(minute=m_bin, second=0, microsecond=0)
            end = start + pd.Timedelta(minutes=30)
            return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"

        filtered["TimeWindow"] = filtered["OpenTime_eat"].apply(time_bin)
        time_stats = filtered.groupby("TimeWindow")["Profit"].sum()

        if not time_stats.empty:
            worst_time_window = time_stats.idxmin()
            worst_pnl = time_stats.min()
            best_time_window = time_stats.idxmax()
            best_pnl = time_stats.max()
        else:
            worst_time_window = best_time_window = None
            worst_pnl = best_pnl = 0

        # -------------------- LEFT: Worst Trade Time Window --------------------
        with t1:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Worst Trade Time Window</div>", unsafe_allow_html=True)
            if worst_time_window:
                color = "#ff5b5b" if worst_pnl < 0 else "#26a269"
                html = f"<div class='insight-card' style='text-align:center'><strong>{worst_time_window}</strong><br>Total P&L: <span style='color:{color}'>${worst_pnl:,.2f}</span></div>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No trade time data.")

        # -------------------- RIGHT: Best Trade Time Window --------------------
        with t2:
            st.markdown("<div style='font-weight:700;margin-bottom:6px'>Best Trade Time Window</div>", unsafe_allow_html=True)
            if best_time_window:
                color = "#26a269" if best_pnl >= 0 else "#ff5b5b"
                html = f"<div class='insight-card' style='text-align:center'><strong>{best_time_window}</strong><br>Total P&L: <span style='color:{color}'>${best_pnl:,.2f}</span></div>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No trade time data.")

    except Exception as e:
        st.info("Not enough data to populate performance summaries.")

else:
    st.info("No trades to compute insights.")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Additional metrics (existing) ----------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><strong>Additional Metrics & Asset Performance</strong><div class='small-muted'>Enhanced overview</div></div>", unsafe_allow_html=True)

if len(filtered) > 0:
    try:
        filtered_sorted = filtered.sort_values("CloseTime")
        equity = filtered_sorted["Profit"].cumsum().reset_index(drop=True)
        running_max = equity.cummax()
        drawdown = running_max - equity
        max_dd = drawdown.max() if len(drawdown)>0 else 0.0
        max_dd_pct = (max_dd / running_max.max() * 100) if (running_max.max() if len(running_max)>0 else 0) != 0 else 0.0
        daily_profit = filtered.groupby(filtered["Date"].dt.date)["Profit"].sum()
        if len(daily_profit) > 1:
            sharpe = (daily_profit.mean() / (daily_profit.std() if daily_profit.std()!=0 else 1)) * np.sqrt(252)
        else:
            sharpe = 0.0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Max Drawdown ($)", f"${max_dd:,.2f}")
        col2.metric("Max Drawdown (%)", f"{max_dd_pct:.2f}%")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Avg Win / Avg Loss", f"${avg_win:.2f} / ${avg_loss:.2f}")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        symbols_ordered = filtered.groupby("Symbol")["Profit"].sum().sort_values(ascending=False).index.tolist()[:6]
        for sym in symbols_ordered:
            sym_df = filtered[filtered["Symbol"]==sym].groupby(filtered["Date"].dt.date).agg(DailyPnL=("Profit","sum")).sort_index()
            sym_cum = sym_df["DailyPnL"].cumsum()
            latest = sym_cum.iloc[-1] if len(sym_cum)>0 else 0
            st.markdown(f"<div style='font-weight:700;margin-top:6px'>{sym} — ${latest:+.2f}</div>", unsafe_allow_html=True)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=sym_cum.index, y=sym_cum.values, mode='lines', line=dict(color="#26a269" if latest>=0 else "#ff5b5b", width=2)))
            fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=60, margin=dict(t=2,b=2,l=10,r=10))
            st.plotly_chart(fig_s, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to compute additional metrics: {e}")
else:
    st.info("No trades to compute additional metrics.")
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#6b7280;font-size:12px'>Genesis — La Khari</div>", unsafe_allow_html=True)













