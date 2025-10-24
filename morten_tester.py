from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf


# ------------------------------
# Konfig
# ------------------------------
TICKER = "^GSPC"   # Benchmark (S&P 500). Bytt til ^W1DOW/URTH/OSEBX hvis ønskelig
RF_ANNUAL = 0.03   # 3% årlig risikofri rente (justér om du vil)


# ------------------------------
# 1) Les fondet og lag avkastning
# ------------------------------
repo_root = Path(__file__).resolve().parent
fund_path = repo_root / "Nordea_stabil_avkastning.csv"
fund = pd.read_csv(fund_path)

# Finn kolonner robust
date_col = next(c for c in fund.columns if "date" in c.lower() or "dato" in c.lower())
price_col = next((c for c in fund.columns if "adj" in c.lower() and "close" in c.lower()), None)
if price_col is None:
    price_col = next(c for c in fund.columns if "close" in c.lower() or "pris" in c.lower() or "kurs" in c.lower())

# Rens og sorter
fund[date_col] = pd.to_datetime(fund[date_col], errors="coerce")
fund = fund.dropna(subset=[date_col, price_col]).sort_values(date_col)
fund = fund.rename(columns={date_col: "Date"})
fund["Fund_Return"] = fund[price_col].astype(float).pct_change()
fund = fund.dropna(subset=["Fund_Return"])

if fund.empty:
    raise ValueError("Fond-data ble tom etter rensing. Sjekk CSV-innholdet.")

# Estimér frekvens: daglig hvis median mellom datoer <= 7 dager, ellers månedlig
freq_days = np.median(np.diff(fund["Date"].values).astype("timedelta64[D]").astype(int))
PER_YEAR = 252 if freq_days <= 7 else 12


# ------------------------------
# 2) Hent benchmark og lag avkastning
# ------------------------------
mkt = yf.download(
    TICKER,
    start=fund["Date"].min(),
    end=fund["Date"].max() + pd.Timedelta(days=1),  # yfinance end er eksklusiv
    progress=False,
    auto_adjust=True,  # gir 'Close' justert for utbytte/splitt
)

# Håndter MultiIndex (kan være ('Close', '^GSPC'))
if isinstance(mkt.columns, pd.MultiIndex):
    mkt = mkt.xs(TICKER, axis=1, level=-1)

# Plukk robust pris-kolonne
if "Close" in mkt.columns:
    price_col_mkt = "Close"
elif "Adj Close" in mkt.columns:
    price_col_mkt = "Adj Close"
else:
    raise ValueError(f"Ingen 'Close' eller 'Adj Close' i market data. Kolonner: {mkt.columns.tolist()}")

mkt = mkt[[price_col_mkt]].rename(columns={price_col_mkt: "Market"}).reset_index(names="Date")

# Merge og lag markedsavkastning
data = pd.merge(fund[["Date", "Fund_Return"]], mkt, on="Date", how="inner").sort_values("Date")
data["Market_Return"] = data["Market"].astype(float).pct_change()
data = data.dropna(subset=["Fund_Return", "Market_Return"])

if data.empty:
    raise ValueError("Ingen overlapp mellom fondet og benchmark på dato. Sjekk dato-intervaller.")


# ------------------------------
# 3) Basismetrikker (uavhengig av benchmark)
# ------------------------------
rf_per = RF_ANNUAL / PER_YEAR

ann_mean = data["Fund_Return"].mean() * PER_YEAR
ann_vol  = data["Fund_Return"].std(ddof=1) * np.sqrt(PER_YEAR)
sharpe   = ((data["Fund_Return"] - rf_per).mean() / data["Fund_Return"].std(ddof=1)) * np.sqrt(PER_YEAR)


# ------------------------------
# 4) CAPM på overskuddsavkastning (excess returns)
#     (r_p - r_f) = alpha + beta (r_m - r_f) + eps
# ------------------------------
y = data["Fund_Return"] - rf_per
x = data["Market_Return"] - rf_per
X = sm.add_constant(x)
capm = sm.OLS(y, X).fit()
alpha_daily = capm.params["const"]
beta = capm.params["Market_Return"]
sigma_e_daily = capm.resid.std(ddof=1)

alpha_annual = alpha_daily * PER_YEAR
sigma_e_annual = sigma_e_daily * np.sqrt(PER_YEAR)


# ------------------------------
# 5) Ytelsesmål
# ------------------------------
# Treynor: årlig overskuddsavkastning per beta
EPS = 1e-12
treynor = (ann_mean - RF_ANNUAL) / (beta if abs(beta) > EPS else np.nan)

# Jensen's alpha (årlig)
mean_mkt_annual = data["Market_Return"].mean() * PER_YEAR
jensen = ann_mean - (RF_ANNUAL + beta * (mean_mkt_annual - RF_ANNUAL))

# Information Ratio (iht. forelesning): IR = alpha / sigma_e
IR = alpha_annual / (sigma_e_annual if sigma_e_annual > EPS else np.nan)

# M^2 = Rf + Sharpe * sigma_M (alle årlig)
mkt_vol_annual = data["Market_Return"].std(ddof=1) * np.sqrt(PER_YEAR)
M2 = RF_ANNUAL + sharpe * mkt_vol_annual


# ------------------------------
# 6) Utskrift (pent formatert)
# ------------------------------
start, end = data["Date"].min().date(), data["Date"].max().date()
print("\n--- Datointervall og frekvens ---")
print(f"Periode: {start} → {end}  | Observasjoner: {len(data)}  | PER_YEAR={PER_YEAR}")

print("\n--- Grunnleggende ---")
print(f"Årlig gj.snitt (aritmetisk):   {ann_mean: .3%}")
print(f"Årlig volatilitet:             {ann_vol: .3%}")
print(f"Sharpe (Rf={RF_ANNUAL:.1%}):   {sharpe: .3f}")

print("\n--- CAPM ---")
print(f"Beta:                          {beta: .4f}")
print(f"Alpha (årlig):                 {alpha_annual: .3%}")
print(f"Residual-vol (årlig):          {sigma_e_annual: .3%}")

print("\n--- Ytelsesmål ---")
print(f"Treynor:                       {treynor: .3f}")
print(f"Jensen's alpha:                {jensen: .3%}")
print(f"Information ratio:             {IR: .3f}")
print(f"M^2:                           {M2: .3%}")
