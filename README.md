# Temperature Trend Analyzer (Climate Change Detection)

This project analyzes temperature and rainfall trends from local weather data using linear regression. It detects climate change signals in your area, visualizes trends, and provides statistical summaries and predictions.

## Features
- Reads weather_data.csv (columns: date, temp_celsius, rainfall_mm, humidity_percent)
- Calculates average summer (Jun-Aug) and winter (Dec-Feb) temperatures per year
- Performs linear regression to detect trends in temperature and rainfall
- Outputs regression equation, slope, correlation coefficient, p-value, and predictions
- Visualizes trends with scatter plots and trend lines
- Summarizes findings in a conclusion

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the analysis: `python analyze_trends.py`

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, scipy
