import yfinance as yf
from pathlib import Path
import pandas as pd

# --- 1) Hent GSPC fra Yahoo Finance ---
mkt = yf.download(
    "^GSPC",
    start="2022-07-03",      # fra dato
    end="2024-08-14",        # til dato
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
