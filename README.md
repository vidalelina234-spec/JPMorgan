# JPMorgan
Quantitative Research
#Task 1) Visualize and analyse the data to find patterns within the market price of natural fas delivered at the end of each calendar month.

# gas_forecast.py -- full end-to-end script with explanations
import warnings
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# -----------------------------
# 1) Paste the monthly data (CSV format)
# -----------------------------
data_str = """Date,Price
10/31/2020,1.01E+01
11/30/2020,1.03E+01
12/31/2020,1.10E+01
1/31/2021,1.09E+01
2/28/2021,1.09E+01
3/31/2021,1.09E+01
4/30/2021,1.04E+01
5/31/2021,9.84E+00
6/30/2021,1.00E+01
7/31/2021,1.01E+01
8/31/2021,1.03E+01
9/30/2021,1.02E+01
10/31/2021,1.01E+01
11/30/2021,1.12E+01
12/31/2021,1.14E+01
1/31/2022,1.15E+01
2/28/2022,1.18E+01
3/31/2022,1.15E+01
4/30/2022,1.07E+01
5/31/2022,1.07E+01
6/30/2022,1.04E+01
7/31/2022,1.05E+01
8/31/2022,1.04E+01
9/30/2022,1.08E+01
10/31/2022,1.10E+01
11/30/2022,1.16E+01
12/31/2022,1.16E+01
1/31/2023,1.21E+01
2/28/2023,1.17E+01
3/31/2023,1.20E+01
4/30/2023,1.15E+01
5/31/2023,1.12E+01
6/30/2023,1.09E+01
7/31/2023,1.14E+01
8/31/2023,1.11E+01
9/30/2023,1.15E+01
10/31/2023,1.18E+01
11/30/2023,1.22E+01
12/31/2023,1.28E+01
1/31/2024,1.26E+01
2/29/2024,1.24E+01
3/31/2024,1.27E+01
4/30/2024,1.21E+01
5/31/2024,1.14E+01
6/30/2024,1.15E+01
7/31/2024,1.16E+01
8/31/2024,1.15E+01
9/30/2024,1.18E+01
"""
# Explanation: The file above is CSV text (Date,Price). Prices are in scientific notation (1.01E+01 = 10.1).

# -----------------------------
# 2) Read into a pandas DataFrame and prepare the series
# -----------------------------
df = pd.read_csv(io.StringIO(data_str), parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # ensure numeric
df.set_index('Date', inplace=True)
monthly = df['Price'].copy()   # 'monthly' is a pandas Series indexed by month-end dates

# Quick check
print("Loaded rows:", len(monthly))
print("Range:", monthly.index.min().date(), "to", monthly.index.max().date())

# -----------------------------
# 3) Create a daily interpolated time series for the historical period
# -----------------------------
start_date = monthly.index.min()
end_date = monthly.index.max()

daily_index_hist = pd.date_range(start=start_date, end=end_date, freq='D')
# reindex to daily index and use time interpolation between known monthly points
daily_from_monthly = monthly.reindex(daily_index_hist).interpolate(method='time')

# Explanation:
# - reindex(...).interpolate(method='time') fills in daily values between the monthly end-of-month points
# - this gives us an estimate for ANY day in the historical window (smooth linear-in-time interpolation)

# -----------------------------
# 4) Build features for modeling: trend + yearly seasonality
# -----------------------------
def make_features(dates, ref_date=None):
    """
    Build a feature matrix for a list of dates:
      - t: months (fractional) since ref_date
      - sin and cos terms for annual seasonality (12-month cycle)
    ref_date: datetime-like. If None, uses the earliest date across the provided dates
    Returns: X (2D numpy array), t (1D float array)
    """
    dates = pd.to_datetime(dates)
    if ref_date is None:
        ref_date = dates.min()
    else:
        ref_date = pd.to_datetime(ref_date)
    # months difference (integer part)
    months_since = (dates.year - ref_date.year) * 12 + (dates.month - ref_date.month)
    # fractional part of the month from the day index (0 = start of month)
    day_fraction = []
    for d in dates:
        days_in_month = pd.Period(d, freq='M').days_in_month
        frac = (d.day - 1) / days_in_month
        day_fraction.append(frac)
    day_fraction = np.array(day_fraction, dtype=float)
    t = months_since.astype(float) + day_fraction
    # seasonal terms using calendar month (1..12). This captures roughly the annual cycle.
    months = dates.month
    sin = np.sin(2 * np.pi * (months / 12.0))
    cos = np.cos(2 * np.pi * (months / 12.0))
    X = np.column_stack([t, sin, cos])
    return X, t

# Why these features?
# - t captures a smooth linear growth or decline over months (trend)
# - sin/cos with period 12 captures repeating yearly patterns (seasonality)
# - combining them gives a simple but effective 'trend + seasonality' model

# -----------------------------
# 5) Train a Ridge regression model on the monthly points
# -----------------------------
X_train, t_train = make_features(monthly.index, ref_date=start_date)
y_train = monthly.values
model = Ridge(alpha=0.5)   # Ridge = linear regression with L2 regularization
# alpha=0.5 is a small regularization term to avoid overfitting noise in a small dataset
model.fit(X_train, y_train)
y_fit = model.predict(X_train)
residuals = y_train - y_fit
resid_std = residuals.std(ddof=1)   # sample standard deviation (ddof=1)

print("Trained Ridge model. Residual std:", round(resid_std, 4))

# -----------------------------
# 6) Forecast 12 months ahead (monthly end-of-month points)
# -----------------------------
future_month_ends = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=12, freq='M')
X_future, t_future = make_features(future_month_ends, ref_date=start_date)  # note ref_date=start_date!
y_future_pred = model.predict(X_future)

# Build an extended monthly series (history + forecast)
monthly_pred_extended = pd.concat([monthly, pd.Series(y_future_pred, index=future_month_ends)])
monthly_pred_extended = monthly_pred_extended.sort_index()

# -----------------------------
# 7) Create a daily curve for the whole extended horizon (history + forecast)
# -----------------------------
daily_index_ext = pd.date_range(start=start_date, end=future_month_ends[-1], freq='D')
# place monthly end-of-month values on the calendar, then interpolate daily
daily_extended = monthly_pred_extended.reindex(daily_index_ext).interpolate(method='time')

# Simple CI band (approximate) using residual std from training
daily_ci_upper = daily_extended + 1.96 * resid_std
daily_ci_lower = daily_extended - 1.96 * resid_std

# -----------------------------
# 8) Function to estimate price for any date (past within history or up to 1-year ahead)
# -----------------------------
first_date = start_date
last_hist_date = end_date
last_forecast_date = future_month_ends[-1]

def estimate_price(date_input):
    """
    Returns {'date': date, 'price': estimate, 'method': ...}
    - If date < first historical date: model extrapolation (less reliable)
    - If date within history or within forecast horizon: returns daily interpolated or forecasted value
    - If date beyond the 1-year forecast: raises ValueError
    """
    date = pd.to_datetime(date_input)
    if date < first_date:
        # extrapolate backward with the model (use same ref_date)
        warnings.warn("Date before history; using model extrapolation (less reliable).", UserWarning)
        Xp, _ = make_features([date], ref_date=start_date)
        price = float(model.predict(Xp)[0])
        return {'date': date.date(), 'price': price, 'method': 'extrapolation_before_history'}
    if date > last_forecast_date:
        raise ValueError(f"Date {date.date()} beyond forecast horizon ({last_forecast_date.date()}).")
    price = float(daily_extended.loc[date])
    return {'date': date.date(), 'price': price, 'method': 'daily_interpolation_or_forecast'}

# -----------------------------
# 9) Plot results (two plots)
# -----------------------------
plt.figure(figsize=(10,5))
plt.scatter(monthly.index, monthly.values, label='Monthly actuals (points)', color='orange')
plt.plot(monthly.index, y_fit, label='Model fit (historical)', linewidth=1)
plt.plot(monthly_pred_extended.index, monthly_pred_extended.values, label='Monthly extended (hist + forecast)', linewidth=1)
plt.title("Monthly natural gas prices: actuals, model fit, and 12-month forecast")
plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(daily_extended.index, daily_extended.values, label='Daily extended curve')
plt.fill_between(daily_extended.index, daily_ci_lower.values, daily_ci_upper.values, alpha=0.2, label='Approx 95% CI')
plt.scatter(monthly.index, monthly.values, label='Monthly actuals', color='orange', s=10)
plt.title("Daily interpolated price (historical + 12-mo forecast)")
plt.xlabel("Date"); plt.ylabel("Price"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 10) Export monthly forecast table to CSV and show example estimates
# -----------------------------
forecast_table = pd.DataFrame({
    'MonthEnd': monthly_pred_extended.index,
    'Price_Estimate': monthly_pred_extended.values,
})
forecast_table['Lower_CI'] = forecast_table['Price_Estimate'] - 1.96 * resid_std
forecast_table['Upper_CI'] = forecast_table['Price_Estimate'] + 1.96 * resid_std

forecast_table.to_csv("monthly_forecast.csv", index=False)
print("Saved monthly_forecast.csv (history + next 12 months).")

# Example calls:
print(estimate_price('2022-03-20'))   # past date example
print(estimate_price(str(last_hist_date.date())))  # last historical date
print(estimate_price(str((last_hist_date + pd.Timedelta(days=45)).date())))  # near future inside horizon
print(estimate_price(str(last_forecast_date.date())))  # last forecast date
