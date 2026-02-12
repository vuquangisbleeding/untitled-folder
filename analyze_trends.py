import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
df = pd.read_csv('weather_data.csv', parse_dates=['date'])
# Drop rows with missing values in relevant columns
df = df.dropna(subset=['date', 'temp_celsius', 'rainfall_mm', 'humidity_percent'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Helper: get seasonal averages

def seasonal_avg(df, months):
    # Drop missing temp_celsius values for the selected months
    return df[df['month'].isin(months)].dropna(subset=['temp_celsius']).groupby('year')['temp_celsius'].mean()


def rainfall_avg(df, months):
    # Drop missing rainfall_mm values for the selected months
    return df[df['month'].isin(months)].dropna(subset=['rainfall_mm']).groupby('year')['rainfall_mm'].mean()

# Summer: Jun-Aug, Winter: Dec-Feb
summer_months = [6, 7, 8]
winter_months = [12, 1, 2]

summer_temps = seasonal_avg(df, summer_months)
winter_temps = seasonal_avg(df, winter_months)
summer_rain = rainfall_avg(df, summer_months)

# Linear regression for summer temps

# Linear regression from scratch (least squares)
def analyze_trend(years, values, label):
    x = np.array(years)
    y = np.array(values)
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Sxy = np.sum((x - x_mean) * (y - y_mean))
    Sxx = np.sum((x - x_mean) ** 2)
    slope = Sxy / Sxx
    intercept = y_mean - slope * x_mean
    # Pearson correlation coefficient
    r_num = np.sum((x - x_mean) * (y - y_mean))
    r_den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r_value = r_num / r_den if r_den != 0 else 0
    # Calculate p-value (two-sided, t-distribution)
    t_stat = r_value * np.sqrt((n - 2) / (1 - r_value ** 2)) if n > 2 and abs(r_value) < 1 else 0
    from scipy.stats import t
    p_value = 2 * t.sf(np.abs(t_stat), df=n - 2) if n > 2 else 1
    # Standard error and confidence intervals for slope and intercept
    y_pred = slope * x + intercept
    residuals = y - y_pred
    residual_std = np.sqrt(np.sum(residuals ** 2) / (n - 2)) if n > 2 else 0
    slope_se = residual_std / np.sqrt(Sxx) if Sxx > 0 and n > 2 else 0
    intercept_se = residual_std * np.sqrt(1/n + x_mean**2/Sxx) if Sxx > 0 and n > 2 else 0
    # 95% confidence intervals
    alpha = 0.05
    t_crit = t.ppf(1 - alpha/2, df=n-2) if n > 2 else 0
    slope_ci = (slope - t_crit * slope_se, slope + t_crit * slope_se)
    intercept_ci = (intercept - t_crit * intercept_se, intercept + t_crit * intercept_se)
    print(f"{label} regression: y = {slope:.3f}x + {intercept:.2f}")
    print(f"Slope: {slope:.3f} per year (95% CI: {slope_ci[0]:.3f} to {slope_ci[1]:.3f})")
    print(f"Intercept: {intercept:.2f} (95% CI: {intercept_ci[0]:.2f} to {intercept_ci[1]:.2f})")
    print(f"Correlation coefficient r = {r_value:.2f}")
    print(f"P-value = {p_value:.4f}")
    return slope, intercept, r_value, p_value

def plot_trend(years, values, slope, intercept, label, ylabel):
    plt.scatter(years, values, label='Avg ' + label)
    plt.plot(years, slope * years + intercept, color='red', label='Trend line')
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(f'{label} Trend')
    plt.legend()
    plt.show()

print('--- Summer Temperature Trend ---')
s_years = summer_temps.index.values
s_values = summer_temps.values
s_slope, s_intercept, s_r, s_p = analyze_trend(s_years, s_values, 'Summer Temp')
plot_trend(s_years, s_values, s_slope, s_intercept, 'Summer Temp', 'Avg Temp (°C)')

# Predict 2030 summer temp
future_year = 2030
pred_temp = s_slope * future_year + s_intercept
print(f"Predicted summer avg temp in {future_year}: {pred_temp:.2f}°C")

print('\n--- Winter Temperature Trend ---')
w_years = winter_temps.index.values
w_values = winter_temps.values
w_slope, w_intercept, w_r, w_p = analyze_trend(w_years, w_values, 'Winter Temp')
plot_trend(w_years, w_values, w_slope, w_intercept, 'Winter Temp', 'Avg Temp (°C)')

print('\n--- Summer Rainfall Trend ---')
r_years = summer_rain.index.values
r_values = summer_rain.values
r_slope, r_intercept, r_r, r_p = analyze_trend(r_years, r_values, 'Summer Rainfall')
plot_trend(r_years, r_values, r_slope, r_intercept, 'Summer Rainfall', 'Avg Rainfall (mm)')

# Summary
print('\n--- Conclusion ---')
print(f"Summers warming: {s_slope*10:.2f}°C per decade.")
print(f"Winters warming: {w_slope*10:.2f}°C per decade.")
print(f"Summer rainfall trend: {r_slope:.2f} mm/year.")
if s_p < 0.05:
    print("Summer temperature trend is statistically significant.")
if w_p < 0.05:
    print("Winter temperature trend is statistically significant.")
if r_p < 0.05:
    print("Rainfall trend is statistically significant.")
