import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

# --- FILBANE TIL FONDET ---
fund_file = r"C:\Users\Eier\OneDrive\Eldig sem. 5\FIN3009\Project_1\Prosjekt_FIN3009\Nordea_stabil_avkastning.csv"

# --- LES CSV ---
fund_data = pd.read_csv(fund_file)

# --- AUTOMATISK FINN KOLONNER ---
date_col = next((col for col in fund_data.columns if 'date' in col.lower() or 'dato' in col.lower()), None)
price_col = next((col for col in fund_data.columns if 'close' in col.lower() or 'pris' in col.lower() or 'kurs' in col.lower()), None)

# --- KONVERTER DATO ---
fund_data[date_col] = pd.to_datetime(fund_data[date_col])
fund_data = fund_data.sort_values(date_col)

# --- HENT BENCHMARK FRA YAHOO FINANCE ---
benchmark_ticker = "^GSPC"  # S&P 500, kan endres til annen indeks
market_data = yf.download(
    benchmark_ticker,
    start=fund_data[date_col].min(),
    end=fund_data[date_col].max(),
    auto_adjust=False
)

# --- Flatten kolonner hvis MultiIndex ---
if isinstance(market_data.columns, pd.MultiIndex):
    market_data.columns = [' '.join(col).strip() for col in market_data.columns.values]

# --- Reset index for å få 'Date' som kolonne ---
market_data = market_data.reset_index()

# --- Finn riktig pris-kolonne robust ---
price_col_market = None
for col in market_data.columns:
    if col.startswith('Adj Close'):
        price_col_market = col
        break
    elif col.startswith('Close'):
        price_col_market = col
        break

if price_col_market is None:
    print("Følgende kolonner finnes i market_data:", market_data.columns)
    raise ValueError("Ingen pris-kolonne funnet i market_data. Sjekk ticker og tidsperiode.")

print(f"Bruker '{price_col_market}' som benchmark-pris-kolonne.")

# --- SLÅ SAMMEN FOND OG BENCHMARK PÅ DATO ---
data = pd.merge(
    fund_data[[date_col, price_col]],
    market_data[['Date', price_col_market]],
    left_on=date_col,
    right_on='Date',
    how='inner'
)
data.rename(columns={price_col: 'Fund', price_col_market: 'Market'}, inplace=True)

# --- BEREGN DAGLIGE AVKASTNINGER ---
data['Fund_Return'] = data['Fund'].pct_change()
data['Market_Return'] = data['Market'].pct_change()
data = data.dropna()

# --- RISIKOFRI RENTE (DAGLIG) ---
risk_free_rate = 0.03 / 252  # 3% årlig delt på handelsdager

# --- BETA OG JENSEN'S ALPHA ---
X = sm.add_constant(data['Market_Return'])
model = sm.OLS(data['Fund_Return'], X).fit()
alpha = model.params['const']
beta = model.params['Market_Return']

# --- SHARPE RATIO ---
excess_returns = data['Fund_Return'] - risk_free_rate
sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

# --- TREYNOR RATIO ---
treynor_ratio = (data['Fund_Return'].mean() - risk_free_rate) / beta * 252

# --- INFORMATION RATIO ---
active_return = data['Fund_Return'] - data['Market_Return']
information_ratio = active_return.mean() / active_return.std() * np.sqrt(252)

# --- M^2 (Modigliani-Modigliani) ---
M2 = risk_free_rate + (excess_returns.mean() / excess_returns.std()) * data['Market_Return'].std() * np.sqrt(252)

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
plt.plot(data[date_col], data['Mean'], label='Glidende gjennomsnitt')
plt.fill_between(
    data[date_col],
    data['Mean'] - data['SE'],
    data['Mean'] + data['SE'],
    alpha=0.2,
    color='blue',
    label='± Standard Error'
)
plt.xlabel('Dato')
plt.ylabel('Daglig avkastning')
plt.title(f'Fondets avkastning med benchmark ({benchmark_ticker})')
plt.legend()
plt.grid(True)
plt.show()