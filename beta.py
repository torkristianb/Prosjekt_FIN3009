import pandas as pd
import numpy as np

fund = pd.read_csv("Nordea_stabil_avkastning.csv")
gspc = pd.read_csv("gspc_download.csv")


fund_date_col = next(c for c in fund.columns if "date" in c.lower() or "dato" in c.lower())
price_col_fund = next(c for c in fund.columns if "close" in c.lower() or "pris" in c.lower())
gspc_date_col = next(c for c in gspc.columns if "date" in c.lower())
price_col_gspc = next(c for c in gspc.columns if "close" in c.lower())

fund[fund_date_col] = pd.to_datetime(fund[fund_date_col], errors="coerce")
gspc[gspc_date_col] = pd.to_datetime(gspc[gspc_date_col], errors="coerce")

# 🔹 Her konverteres tekst til tall
fund[price_col_fund] = pd.to_numeric(fund[price_col_fund], errors="coerce")
gspc[price_col_gspc] = pd.to_numeric(gspc[price_col_gspc], errors="coerce")

fund = fund.dropna(subset=[fund_date_col, price_col_fund]).sort_values(fund_date_col)
gspc = gspc.dropna(subset=[gspc_date_col, price_col_gspc]).sort_values(gspc_date_col)

fund["Fund_Return"] = fund[price_col_fund].pct_change()
gspc["Market_Return"] = gspc[price_col_gspc].pct_change()

data = pd.merge(
    fund[[fund_date_col, "Fund_Return"]],
    gspc[[gspc_date_col, "Market_Return"]],
    left_on=fund_date_col,
    right_on=gspc_date_col,
    how="inner"
).dropna(subset=["Fund_Return", "Market_Return"])

cov = np.cov(data["Fund_Return"], data["Market_Return"], ddof=1)[0, 1]
var_mkt = np.var(data["Market_Return"], ddof=1)
beta = cov / var_mkt

print(f"Beta = {beta:.4f}")
