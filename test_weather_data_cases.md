# Test Cases for Temperature Trend Analyzer

Below are 6 test cases (including edge cases) for `weather_data.csv` to validate the analysis script:

---
## 1. Normal Data (Multiple Years, No Missing)
| date       | temp_celsius | rainfall_mm | humidity_percent |
|------------|--------------|-------------|------------------|
| 2000-06-15 | 25.0         | 50          | 60               |
| 2000-07-15 | 26.0         | 55          | 62               |
| 2001-06-15 | 26.5         | 48          | 61               |
| 2001-07-15 | 27.0         | 52          | 63               |
| 2002-06-15 | 27.5         | 47          | 60               |
| 2002-07-15 | 28.0         | 49          | 64               |

---
## 2. All Values Missing in a Year (Edge)
| date       | temp_celsius | rainfall_mm | humidity_percent |
|------------|--------------|-------------|------------------|
| 2000-06-15 |              | 50          | 60               |
| 2000-07-15 |              | 55          | 62               |
| 2001-06-15 | 26.5         | 48          | 61               |
| 2001-07-15 | 27.0         | 52          | 63               |

---
## 3. Only One Year of Data (Edge)
| date       | temp_celsius | rainfall_mm | humidity_percent |
|------------|--------------|-------------|------------------|
| 2010-06-15 | 24.0         | 60          | 70               |
| 2010-07-15 | 25.0         | 62          | 72               |
| 2010-08-15 | 26.0         | 61          | 71               |

---
## 4. Constant Temperature (No Trend)
| date       | temp_celsius | rainfall_mm | humidity_percent |
|------------|--------------|-------------|------------------|
| 2015-06-15 | 22.0         | 40          | 60               |
| 2016-06-15 | 22.0         | 41          | 61               |
| 2017-06-15 | 22.0         | 39          | 62               |
| 2018-06-15 | 22.0         | 42          | 63               |

---
## 5. Perfect Linear Increase
| date       | temp_celsius | rainfall_mm | humidity_percent |
|------------|--------------|-------------|------------------|
| 2000-06-15 | 20.0         | 30          | 50               |
| 2001-06-15 | 21.0         | 31          | 51               |
| 2002-06-15 | 22.0         | 32          | 52               |
| 2003-06-15 | 23.0         | 33          | 53               |

---
## 6. Outlier Year (Extreme Value)
| date       | temp_celsius | rainfall_mm | humidity_percent |
|------------|--------------|-------------|------------------|
| 2010-06-15 | 25.0         | 50          | 60               |
| 2011-06-15 | 26.0         | 51          | 61               |
| 2012-06-15 | 100.0        | 52          | 62               |
| 2013-06-15 | 27.0         | 53          | 63               |

---

You can copy-paste these tables (with headers) into `weather_data.csv` to test each scenario. For missing values, leave the cell empty.