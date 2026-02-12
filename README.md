# Temperature Trend Analyzer (Climate Change Detection)

This project analyzes temperature and rainfall trends from local weather data using linear regression. It detects climate change signals in your area, visualizes trends, and provides statistical summaries and predictions.

## Features
- Reads `weather_data.csv` (columns: `date`, `temp_celsius`, `rainfall_mm`, `humidity_percent`)
- Calculates average summer (Jun-Aug) and winter (Dec-Feb) temperatures per year
- Performs linear regression to detect trends in temperature and rainfall
- Outputs regression equation, slope, correlation coefficient, p-value, confidence intervals, and predictions
- Visualizes trends with scatter plots, trend lines, confidence bands, and residuals plots
- Summarizes findings in a conclusion

## Mathematical Details

### 1. Linear Regression (Least Squares)
Given $n$ data points $(x_i, y_i)$, the best-fit line $y = a x + b$ is found by minimizing the sum of squared residuals:

$$
a = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$
$$
b = \bar{y} - a \bar{x}
$$
where $\bar{x}$ and $\bar{y}$ are the means of $x$ and $y$.

### 2. Correlation Coefficient ($r$)
Measures the strength and direction of the linear relationship:
$$
r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}}
$$

### 3. Statistical Significance (p-value)
The $t$-statistic for $r$ is:
$$
t = r \sqrt{\frac{n-2}{1 - r^2}}
$$
The p-value is computed as $2P(T > |t|)$ where $T$ follows the $t$-distribution with $n-2$ degrees of freedom. A small p-value ($<0.05$) indicates a statistically significant trend.

### 4. Confidence Intervals
The standard error of the slope $a$ is:
$$
SE_a = \frac{s}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$
where $s$ is the residual standard deviation. The 95% confidence interval for $a$ is:
$$
a \pm t^* SE_a
$$
where $t^*$ is the critical value from the $t$-distribution.

### 5. Residuals
Residuals are $e_i = y_i - (a x_i + b)$. The residuals plot helps diagnose model fit and outliers.

### 6. Confidence Bands
The 95% confidence band for the regression line at $x$ is:
$$
\hat{y} \pm t^* s \sqrt{\frac{1}{n} + \frac{(x - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the analysis: `python analyze_trends.py`

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, scipy
