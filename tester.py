import yfinance as yf
from pathlib import Path
import pandas as pd

# --- 1) Hent GSPC fra Yahoo Finance ---
mkt = yf.download(
    "^GSPC",
    start="2022-03-07",      # fra dato
    end="2024-08-15",        # til dato
    auto_adjust=True,        # juster for utbytte/splitt
    progress=False
)

# --- 2) Lag avkastningskolonne ---
mkt = mkt.reset_index()  # flytter 'Date' til vanlig kolonne
mkt["Market_Return"] = mkt["Close"].pct_change()

# --- 3) Lagre til CSV i samme mappe ---
out_dir = Path(__file__).resolve().parent
mkt.to_csv(out_dir / "gspc_download.csv", index=False)

# --- 4) Print bekreftelse ---
print(f"\n✅ Lagret: {out_dir/'osebx_download.csv'}")
print(f"Antall rader: {len(mkt)} ({mkt['Date'].min().date()} → {mkt['Date'].max().date()})")
print("\nFørste 5 rader:\n", mkt.head())

import pandas as pd

# --- Les begge datasett ---
fund = pd.read_csv("Nordea_stabil_avkastning.csv")
gspc = pd.read_csv("gspc_download.csv")

# --- Gjett riktig kolonnenavn (for dato og pris) ---
fund_date_col = next(c for c in fund.columns if "date" in c.lower() or "dato" in c.lower())
gspc_date_col = next(c for c in gspc.columns if "date" in c.lower())

# --- Konverter datoer til datetime ---
fund[fund_date_col] = pd.to_datetime(fund[fund_date_col], errors="coerce")
gspc[gspc_date_col] = pd.to_datetime(gspc[gspc_date_col], errors="coerce")

# --- Finn datoer som kun finnes i fondet ---
unique_fund_dates = fund[fund_date_col].dropna().unique()
unique_gspc_dates = gspc[gspc_date_col].dropna().unique()

missing_in_gspc = sorted(set(unique_fund_dates) - set(unique_gspc_dates))

print(f"🔍 Antall datoer i fondet men ikke i GSPC: {len(missing_in_gspc)}")
if len(missing_in_gspc) > 0:
    print("\nEksempler på manglende datoer:")
    print(missing_in_gspc[:10])
