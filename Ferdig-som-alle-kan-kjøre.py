import numpy as np
import pandas as pd
from pathlib import Path

# --------- FINN FILENE ROBUST ---------
def resolve_path(primary: Path, fallbacks: list[Path]) -> Path:
    if primary.exists():
        return primary
    for p in fallbacks:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Fant ikke filen.\nPrøvde:\n - {primary}\n" +
        "".join([f" - {p}\n" for p in fallbacks])
    )

repo_root = Path(__file__).resolve().parent

# Primært: se etter filene i samme mappe som skriptet
fond_prim = repo_root / "Nordea_stabil_avkastning.csv"
mkt_prim  = repo_root / "gspc_download.csv"

# Fallback: din OneDrive-sti (samme som du brukte tidligere)
one = Path.home() / "OneDrive" / "Eldig sem. 5" / "FIN3009" / "Project_1" / "Prosjekt_FIN3009"
fond_fb = one / "Nordea_stabil_avkastning.csv"
mkt_fb  = one / "gspc_download.csv"

fond_fil = resolve_path(fond_prim, [fond_fb])
marked_fil = resolve_path(mkt_prim, [mkt_fb])

# --------- HERFRA ER KODEN DIN UENDRET ---------
# Les inn data
dfp = pd.read_csv(fond_fil)
dfm = pd.read_csv(marked_fil)

# Konverter 'Close' til numerisk og fyll manglende verdier fremover
dfp['Close'] = pd.to_numeric(dfp['Close'], errors='coerce').ffill()
dfm['Close'] = pd.to_numeric(dfm['Close'], errors='coerce').ffill()

# Konverter dato til datetime
dfp['Date'] = pd.to_datetime(dfp['Date'], errors='coerce', format='%m/%d/%Y')
dfm['Date'] = pd.to_datetime(dfm['Date'], errors='coerce', format='%Y-%m-%d')

# Slå sammen på felles datoer
merged = pd.merge(dfm[['Date', 'Close']], dfp[['Date', 'Close']], on='Date', suffixes=('_m', '_p'))
merged = merged.dropna()

print(merged)

# Beregn daglige log-avkastninger
merged['r_m'] = np.log(merged['Close_m'] / merged['Close_m'].shift(1))
merged['r_p'] = np.log(merged['Close_p'] / merged['Close_p'].shift(1))
merged = merged.dropna()

# Parametre
rf = 0.03  # Risikofri rente
N = len(merged)
trading_days = 575  # Antall handelsdager per år (lar denne stå som du hadde)
#trading_days = 252
# Daglig gjennomsnitt og varians
rbar_p = merged['r_p'].mean()
rbar_m = merged['r_m'].mean()
var_p = merged['r_p'].var(ddof=1)
var_m = merged['r_m'].var(ddof=1)

# Beta
cov_pm = np.cov(merged['r_p'], merged['r_m'], ddof=1)[0, 1]
beta = cov_pm / var_m

# Årlige log-avkastninger (annualisert)
rp_annual = rbar_p * trading_days
rm_annual = rbar_m * trading_days

# Årlig standardavvik (volatilitet)
sigma_p_annual = np.sqrt(var_p * trading_days)
sigma_m_annual = np.sqrt(var_m * trading_days)

# Jensen's alpha
jensen_alpha = rp_annual - rf - beta * (rm_annual - rf)

# Sharpe ratio
sharpe_p = (rp_annual - rf) / sigma_p_annual
sharpe_m = (rm_annual - rf) / sigma_m_annual

# Treynor ratio
treynor_p = (rp_annual - rf) / beta

# M-square
scaling = sigma_p_annual / sigma_m_annual
M2 = scaling * rp_annual + (1 - scaling) * rf - rm_annual
#M2 = rf + sharpe_p * sigma_m_annual

# IR
tracking_error_annual = np.sqrt(np.var(merged['r_p'] - merged['r_m'], ddof=1) * trading_days)
information_ratio = (rp_annual - rm_annual) / tracking_error_annual
# Print resultater
print("Daglig log-avkastning:")
print("Fond:", rbar_p*100, "%")
print("Marked:", rbar_m*100, "%")
print("\nÅrlig:")
print("Nordea log-avkastning:", rp_annual*100, "%")
print("Marked log-avkastning:", rm_annual*100, "%")
print("Nordea volatilitet:", sigma_p_annual*100, "%")
print("Marked volatilitet:", sigma_m_annual*100, "%")
print("\nBeta:", beta)
print("Jensen's alpha:", jensen_alpha)
print("Sharpe ratio (fond, marked):", (sharpe_p, sharpe_m))
print("Treynor ratio:", treynor_p)
print("M-square:", M2)
print("Information ratio:", information_ratio)

