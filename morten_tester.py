from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

# --- FILBANE TIL FONDET (relativ til skriptet) ---
repo_root = Path(__file__).resolve().parent
fund_file = repo_root / "Nordea_stabil_avkastning.csv"

# --- LES CSV ---
fund_data = pd.read_csv(fund_file)

# --- AUTOMATISK FINN KOLONNER ---
date_col = next((c for c in fund_data.columns if 'date' in c.lower() or 'dato' in c.lower()), None)
price_col = next((c for c in fund_data.columns if 'adj' in c.lower() and 'close' in c.lower()), None)
if price_col is None:
    price_col = next((c for c in fund_data.columns if 'close' in c.lower() or 'pris' in c.lower() or 'kurs' in c.lower()), None)
if date_col is None or price_col is None:
    raise ValueError(f"Fant ikke dato/pris-kolonner i fond-data. Kolonner: {list(fund_data.columns)}")

# --- KONVERTER DATO & sorter ---
fund_data[date_col] = pd.to_datetime(fund_data[date_col], errors="coerce")
fund_data = fund_data.dropna(subset=[date_col, price_col]).sort_values(date_col)

# --- HENT BENCHMARK FRA YAHOO FINANCE ---
benchmark_ticker = "^GSPC"  # S&P 500
market_data = yf.download(
    benchmark_ticker,
    start=fund_data[date_col].min(),
    end=fund_data[date_col].max() + pd.Timedelta(days=1),  # yfinance end er eksklusiv
    auto_adjust=False,
    progress=False
)

# --- Håndter MultiIndex/single index ---
if isinstance(market_data.columns, pd.MultiIndex):
    # ta ut nivå for tickeren hvis MultiIndex (felt, ticker)
    market_data = market_data.xs(benchmark_ticker, axis=1, level=-1)

market_data = market_data.reset_index()

# --- Finn riktig pris-kolonne robust ---
price_col_market = "Adj Close" if "Adj Close" in market_data.columns else ("Close" if "Close" in market_data.columns else None)
if price_col_market is None:
    raise ValueError(f"Ingen pris-kolonne funnet i market_data. Kolonner: {list(market_data.columns)}")

print(f"Bruker '{price_col_market}' som benchmark-pris-kolonne.")

# --- SLÅ SAMMEN FOND OG BENCHMARK PÅ DATO ---
left = fund_data[[date_col, price_col]].rename(columns={date_col: "Date", price_col: "Fund"})
right = market_data[["Date", price_col_market]].rename(columns={price_col_market: "Market"})
data = pd.merge(left, right, on="Date", how="inner").sort_values("Date")

# --- BEREGN DAGLIGE AVKASTNINGER (samme funksjonalitet) ---
data['Fund_Return'] = data['Fund'].pct_change()
data['Market_Return'] = data['Market'].pct_change()
data = data.dropna()

# --- RISIKOFRI RENTE (DAGLIG) ---
risk_free_rate = 0.03 / 252  # 3% årlig delt på handelsdager

# --- BETA OG JENSEN'S ALPHA (samme modell som før) ---
X = sm.add_constant(data['Market_Return'])
model = sm.OLS(data['Fund_Return'], X).fit()
alpha = model.params['const']
beta = model.params['Market_Return']

# --- SHARPE RATIO ---
excess_returns = data['Fund_Return'] - risk_free_rate
sharpe_ratio = excess_returns.mean() / (excess_returns.std() + 1e-12) * np.sqrt(252)

# --- TREYNOR RATIO ---
treynor_ratio = (data['Fund_Return'].mean() - risk_free_rate) / (beta + 1e-12) * 252

# --- INFORMATION RATIO ---
active_return = data['Fund_Return'] - data['Market_Return']
information_ratio = active_return.mean() / (active_return.std() + 1e-12) * np.sqrt(252)

# --- M^2 (beholder samme uttrykk som dere brukte) ---
M2 = risk_free_rate + (excess_returns.mean() / (excess_returns.std() + 1e-12)) * data['Market_Return'].std() * np.sqrt(252)

# --- RESULTATER ---
print(f"Jensen's alpha: {alpha*252:.4f}")
print(f"Beta: {beta:.4f}")
print(f"Sharpe ratio: {sharpe_ratio:.4f}")
print(f"Treynor ratio: {treynor_ratio:.4f}")
print(f"Information ratio: {information_ratio:.4f}")
print(f"M^2: {M2:.4f}")

# --- PLOT GLIDENDE AVKASTNING MED STANDARD ERROR ---
window = 10
data['Mean'] = data['Fund_Return'].rolling(window=window).mean()
data['Std'] = data['Fund_Return'].rolling(window=window).std()
data['SE'] = data['Std'] / np.sqrt(window)

plt.figure(figsize=(12,6))
plt.plot(data["Date"], data['Mean'], label='Glidende gjennomsnitt')
plt.fill_between(
    data["Date"],
    data['Mean'] - data['SE'],
    data['Mean'] + data['SE'],
    alpha=0.2,
    label='± Standard Error'
)
plt.xlabel('Dato')
plt.ylabel('Daglig avkastning')
plt.title(f'Fondets avkastning med benchmark ({benchmark_ticker})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
