# Smart Electricity Consumption Analysis and Prediction

**Course:** Data Science and Artificial Intelligence  
**Institution:** Department of Information Technology, National Institute of Technology Karnataka  
**Guide:** Dr. Suresan Pareth  
**Term:** August-December 2026

**Submitted by**

- Ishan Ahirwar (261AI027)
- Arjun S M (261AI012)
- Neelesh Reddy (261AI036)
- Shuban Kamath (261AI058)
- Mourya G N (261AI040)
- Srujan Soraganvi (261AI060)

> The formatted report with figures, certificate, declaration and acknowledgement pages is in `README_Pixel.pdf`. This file holds the same content as plain Markdown (figures are in the PDF).

## Abstract

Electricity consumption varies with time, weather, household requirements and industrial activity. Analysing historical usage helps us understand consumption patterns and estimate future demand. This project analyses 5,000 half-hourly smart-meter readings (1 January to 14 April 2024) using statistical and machine learning techniques.

The data are checked for missing values, duplicates and timing gaps, summarised with descriptive statistics, and examined for correlations. Regression, time-series and anomaly-detection methods are compared against simple baselines, and results are visualised as daily, weekly and monthly patterns.

Only the average past consumption is meaningfully related to consumption (r = 0.32); weather variables are essentially uncorrelated. Linear Regression is the best regression model (R² = 0.24, RMSE 13% below the mean baseline), and a supervised classifier detects labelled anomalies with an F1-score of 0.92.

Because that dataset is short and appears synthetic, the same models are repeated on a larger real-world dataset of 393,440 five-minute power-demand readings (2021-2024) with weather. There, temperature is clearly related to demand (r = 0.47), demand follows strong daily and seasonal cycles, and the models beat the “no change” persistence forecast by 9% (5 minutes ahead) and 26% (30 minutes ahead). The work supports energy monitoring, consumption planning and efficient energy management.

**Keywords:** Electricity Consumption, Energy Analytics, Data Science, Machine Learning, Regression, Time-Series Prediction, Power Demand, Energy Management

## Contents

1. Introduction
2. Dataset Description and Representation
3. Data Preprocessing and Statistical Analysis
4. Proposed Methodology
5. Implementation
6. Results and Discussion
7. Second Dataset and Model Comparison
8. AI Tools and Ethical Considerations
9. Conclusion and Future Scope

Bibliography

## 1. Introduction

### 1.1 Background

Electricity use varies with the time of day, the weather, household needs and industrial activity. Historical usage records help us understand consumption patterns and estimate future demand. Typical applications are energy monitoring, consumption planning and efficient energy management.

### 1.2 Problem Statement

Smart meters log readings every 30 minutes, but raw logs alone do not reveal patterns. Weather, past usage and calendar effects may all matter, yet which of them actually drive consumption is unclear. A data-driven approach is needed to predict consumption and to flag abnormal readings.

### 1.3 Objectives

1. To analyse electricity consumption using descriptive statistics.
2. To preprocess the smart-meter dataset.
3. To investigate correlations between consumption, weather and past usage.
4. To build regression and time-series models to predict consumption.
5. To visualise daily, weekly and monthly patterns and detect anomalies.

### 1.4 Scope of the Project

- **Datasets:** Kaggle Smart Meter Electricity Consumption Dataset (ziya07) and a larger five-minute power-demand dataset with weather.
- **Coverage:** 5,000 half-hourly readings, 1 January to 14 April 2024 (location not specified); 393,440 five-minute readings, 1 January 2021 to 12 December 2024.
- **Parameters:** Consumption or demand, temperature, humidity, wind, pressure and past consumption.
- **Approach:** Software-only analysis; IoT metering is left as a future extension.

## 2. Dataset Description and Representation

### 2.1 Data Collection

The data come from the Kaggle Smart Meter Electricity Consumption Dataset (ziya07). It contains **5,000 observations** and **7 columns**, recorded every **30 minutes** from 1 January to 14 April 2024. A second, larger dataset is described in Chapter 7.

### 2.2 Dataset Features

**Table 2.1: Smart-meter dataset features. All numeric columns are min-max scaled, so values are relative, not kWh.**

| Feature | Type | Description |
|---|:---:|:---:|
| Timestamp | Datetime | Reading time, every 30 minutes |
| Electricity_Consumed | Numerical (0-1) | Target: relative electricity consumption |
| Temperature | Numerical (0-1) | Scaled ambient temperature |
| Humidity | Numerical (0-1) | Scaled relative humidity |
| Wind_Speed | Numerical (0-1) | Scaled wind speed |
| Avg_Past_Consumption | Numerical (0-1) | Average past consumption level |
| Anomaly_Label | Categorical | Normal / Abnormal (250 abnormal readings) |

### 2.3 Tabular Representation

Rows correspond to meter readings and columns to measured features. The first five rows of the dataset are shown below.

**Table 2.2: First five rows of the dataset.**

| Timestamp | Electricity | Temperature | Humidity | Wind speed | Avg past | Label |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 2024-01-01 00:00 | 0.458 | 0.470 | 0.396 | 0.445 | 0.692 | Normal |
| 2024-01-01 00:30 | 0.352 | 0.466 | 0.451 | 0.459 | 0.540 | Normal |
| 2024-01-01 01:00 | 0.483 | 0.285 | 0.408 | 0.470 | 0.615 | Normal |
| 2024-01-01 01:30 | 0.629 | 0.482 | 0.512 | 0.576 | 0.757 | Normal |
| 2024-01-01 02:00 | 0.336 | 0.625 | 0.672 | 0.373 | 0.674 | Normal |

### 2.4 Matrix Representation

The numeric part of the dataset is represented by a matrix, where m is the number of samples and n the number of features. Here **m = 5,000** samples (rows) and **n = 5** numeric features (columns): consumption, temperature, humidity, wind speed and average past consumption.

```
X = [ x_ij ]   (i = 1..m, j = 1..n)
```

## 3. Data Preprocessing and Statistical Analysis

### 3.1 Data Cleaning

The dataset is already clean. The checks and the defensive cleaning applied in code are summarised below.

- **Missing values:** 0. All 7 columns are complete.
- **Duplicates:** 0. There are no duplicate rows or timestamps.
- **Timing:** all 4,999 time steps are exactly 30 minutes apart.
- **Abnormal readings:** 250 readings are labelled Abnormal. They are kept, not removed, because they are the anomaly-detection target.

> **Defensive cleaning applied in code:** sort by timestamp, drop duplicate timestamps, interpolate any gaps, then add the Hour, DayOfWeek, Month and IsWeekend features.

### 3.2 Data Transformation and Normalization

All five numeric columns already lie in [0, 1] (minimum 0, maximum 1), so values show relative consumption, not kWh. No further scaling was applied. For reference, min-max normalization is:

```
x' = (x - x_min) / (x_max - x_min)
```

Derived features added for modelling:

- **Calendar:** Hour, DayOfWeek, Month, IsWeekend.
- **History:** Lag_1, Lag_2, Lag_48, Roll_Mean_6. Lag_48 is consumption 24 hours earlier (48 half-hour steps).
- **Anomaly:** Anomaly (0/1), Roll_Std_6, Diff_From_Avg.

### 3.3 Descriptive Statistics

**Table 3.1: Descriptive statistics of the numeric variables.**

| Variable | Mean | Median | Std. dev. | Variance | Min | Max |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Electricity_Consumed | 0.377 | 0.377 | 0.164 | 0.0269 | 0.000 | 1.000 |
| Temperature | 0.525 | 0.524 | 0.136 | 0.0184 | 0.000 | 1.000 |
| Humidity | 0.498 | 0.497 | 0.147 | 0.0215 | 0.000 | 1.000 |
| Wind_Speed | 0.465 | 0.465 | 0.120 | 0.0145 | 0.000 | 1.000 |
| Avg_Past_Consumption | 0.459 | 0.458 | 0.147 | 0.0215 | 0.000 | 1.000 |

The mode is omitted because, for continuous scaled values, it carries no useful information. Consumption has mean 0.377 and median 0.377, so its distribution is centred and balanced. Mean consumption is almost equal for Normal (0.377) and Abnormal (0.379) readings, but the standard deviation doubles: 0.152 for Normal versus 0.319 for Abnormal.

### 3.4 Correlation Analysis

The Pearson correlation between two variables X and Y is:

```
r_XY = Cov(X, Y) / (sigma_X * sigma_Y)
```

**Table 3.2: Pearson correlation matrix (5,000 readings).**

|   | Consumption | Temperature | Humidity | Wind speed | Avg past |
|---|:---:|:---:|:---:|:---:|:---:|
| Consumption | 1.00 | 0.00 | 0.00 | 0.00 | 0.32 |
| Temperature | 0.00 | 1.00 | -0.01 | -0.01 | 0.00 |
| Humidity | 0.00 | -0.01 | 1.00 | 0.01 | 0.01 |
| Wind speed | 0.00 | -0.01 | 0.01 | 1.00 | 0.00 |
| Avg past | 0.32 | 0.00 | 0.01 | 0.00 | 1.00 |

*Figure 3.1: Pearson correlation heatmap. (chart in README_Pixel.pdf)*

**Avg_Past_Consumption is the only meaningful relationship (r = 0.32).** With |r| < 0.01, temperature, humidity and wind speed are essentially uncorrelated with consumption, so the weather columns carry little predictive signal in this dataset.

## 4. Proposed Methodology

### 4.1 System Workflow

*Figure 4.1: System workflow from data collection to interpretation. (chart in README_Pixel.pdf)*

1. **Data collection:** smart-meter CSV with 5,000 half-hourly rows.
2. **Data preprocessing:** sort, de-duplicate, interpolate and add time features.
3. **Statistical analysis:** summary statistics and correlations.
4. **Feature selection:** weather, history, lag and calendar inputs.
5. **Model development:** regression, forecasting and anomaly detection.
6. **Model evaluation:** MAE, RMSE and R²; precision, recall and F1.
7. **Visualization and interpretation:** patterns, forecasts and anomalies.

> **Evaluation principle:** every model is compared with a simple baseline, and the train/test split is chronological, so models are always tested on readings that come after the data they learned from.

### 4.2 Machine Learning Algorithms

**Table 4.1: Algorithms used for each task.**

| Task | Algorithm | Idea |
|---|:---:|:---:|
| Regression | Linear / Ridge | Weighted sum of features; Ridge penalises large weights |
| Regression | Random Forest | Average of 200 decision trees |
| Regression | Gradient Boosting | Small trees that correct earlier errors |
| Regression | Baseline | Predict the training mean |
| Time series | Naive | Repeat the value from the same time yesterday |
| Time series | ARIMA (2,0,1) | Models autocorrelation and short-term shocks |
| Time series | Decomposition | Checks for a daily cycle (period = 48) |
| Time series | Baseline | Predict the training mean |
| Anomaly detection | Isolation Forest | Unsupervised; anomalies are isolated in fewer random splits |
| Anomaly detection | Random Forest classifier | Supervised; learns from labels with balanced class weights |

Anomaly detection is judged by precision, recall and F1-score, not by accuracy alone.

### 4.3 Feature Engineering

- **Weather:** Temperature, Humidity, Wind_Speed.
- **History:** Avg_Past_Consumption, Lag_1 (30 min), Lag_2 (1 h), Lag_48 (24 h), Roll_Mean_6.
- **Calendar:** Hour, DayOfWeek, IsWeekend.

Roll_Mean_6 is the mean of the previous six readings, shifted by one step so that it never uses the current value.

### 4.4 Model Training and Testing

The split is **chronological 80 / 20**, so no future information leaks into training. The **training set** has 3,961 rows (2 January to 24 March 2024) and the **test set** has 991 rows (24 March to 14 April 2024).

## 5. Implementation

### 5.1 Software and Tools

- **Language:** Python 3 in Jupyter Notebook.
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn.
- **Machine learning:** Scikit-learn.
- **Time series:** Statsmodels.
- **AI assistance:** ChatGPT / Gemini / Copilot (optional).
- **Hardware:** Laptop, 4 GB RAM or more; no sensors needed.

### 5.2 Implementation Procedure

1. Load the CSV and inspect shape, types and ranges.
2. Clean: sort, de-duplicate and interpolate.
3. Engineer calendar, lag and rolling features.
4. Compute statistics, correlations and plots.
5. Train and compare models on the 80% split.
6. Evaluate on the 20% test set and interpret the results.

To run the analysis, place `smart_meter_data.csv` in the same folder as the notebook `electricity_consumption_analysis.ipynb`, install the dependencies with `pip install numpy pandas matplotlib seaborn scikit-learn statsmodels`, and run all cells.

The power-demand analysis in Chapter 7 uses `power_demand_weather_analysis_compare.ipynb` with the same libraries, seed (42) and chronological 80/20 split, so the results are directly comparable.

### 5.3 Sample Code

```python
df = pd.read_csv("smart_meter_data.csv",
                 parse_dates=["Timestamp"])
df = df.sort_values("Timestamp")

print(df.isnull().sum())   # missing values
print(df.describe())       # statistics
print(df.select_dtypes("number").corr())

for lag in [1, 2, 48]:     # 30 min, 1 h, 24 h
    df[f"Lag_{lag}"] = df["Electricity_Consumed"].shift(lag)

split = int(len(df) * 0.8) # chronological split
```

- `sort_values` keeps readings in time order, which the lags and the split rely on.
- `corr()` builds the correlation matrix used in Chapter 3.
- `shift(lag)` copies earlier consumption into the current row as a predictor.
- The `0.8` split makes the test set strictly later than the training set.

## 6. Results and Discussion

### 6.1 Statistical Results

Consumption is nearly flat. Hourly means range from 0.357 to 0.398, weekday means from 0.366 to 0.385, and monthly means from 0.370 to 0.383. There is no strong daily or weekly seasonality, unlike real household data, which is consistent with a synthetic dataset. The descriptive statistics and correlations are in Tables 3.1 and 3.2.

### 6.2 Data Visualization

*Figure 6.1: Average consumption by hour of day. (chart in README_Pixel.pdf)*

*Figure 6.2: Monthly mean consumption (2024). (chart in README_Pixel.pdf)*

Both charts start at zero on purpose, so the flat profile is shown at its true scale.

### 6.3 Model Evaluation

Regression models are scored with MAE, RMSE and R². Anomaly detectors are scored with accuracy, precision, recall, F1-score and a confusion matrix:

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 * Precision * Recall / (Precision + Recall)
```

**Regression.** All models are tested on the chronological test set of 991 readings.

**Table 6.1: Regression results on the test set.**

| Model | MAE | RMSE | R² |
|---|:---:|:---:|:---:|
| Mean baseline | 0.130 | 0.162 | -0.004 |
| Linear Regression | 0.113 | 0.142 | 0.236 |
| Ridge | 0.114 | 0.142 | 0.232 |
| Random Forest | 0.117 | 0.145 | 0.195 |
| Gradient Boosting | 0.116 | 0.143 | 0.217 |

*Figure 6.3: Test RMSE by model (lower is better). (chart in README_Pixel.pdf)*

> **Linear Regression is best:** R² = 0.24 and RMSE 13% lower than the mean baseline. More complex models do not help because the signal is weak.

**Feature importance.** About 52% of the Random Forest importance comes from Avg_Past_Consumption (0.31) and Roll_Mean_6 (0.22). Temperature, humidity and wind speed contribute about 7% each, a small share spread over noisy inputs. IsWeekend contributes about 0.3%, so weekends do not change consumption here. Weather importance does not imply signal, because the correlations were near zero.

*Figure 6.4: Random Forest feature importance. (chart in README_Pixel.pdf)*

**Time-series forecasting.** The last 7 days (336 readings) were forecast with two baselines.

**Table 6.2: Forecast error on the last 7 days (336 readings).**

| Baseline | MAE | RMSE |
|---|:---:|:---:|
| Mean baseline | 0.130 | 0.163 |
| Naive (yesterday) | 0.188 | 0.233 |

*Figure 6.5: Forecast error of the two baselines. (chart in README_Pixel.pdf)*

Yesterday is a poor forecast: RMSE rises from 0.163 (mean) to 0.233 when we repeat the same time yesterday. Consecutive days are almost independent, matching the flat hourly profile. ARIMA(2,0,1) and seasonal decomposition are implemented in the notebook (they require statsmodels) but their results are not reported here.

**Anomaly detection.** The chronological test set has 999 readings (944 Normal, 55 Abnormal).

**Table 6.3: Anomaly-detection metrics for the Abnormal class.**

| Model | Precision | Recall | F1-score |
|---|:---:|:---:|:---:|
| Isolation Forest (unsupervised) | 0.44 | 0.36 | 0.40 |
| Random Forest classifier (supervised) | 0.88 | 0.96 | 0.92 |

**Table 6.4: Confusion matrices (counts of test readings).**

| Model | Actual class | Pred. Normal | Pred. Abnormal |
|---|:---:|:---:|:---:|
| Isolation Forest | Normal | 919 | 25 |
| Isolation Forest | Abnormal | 35 | 20 |
| Random Forest classifier | Normal | 937 | 7 |
| Random Forest classifier | Abnormal | 2 | 53 |

> **Accuracy is misleading:** always predicting Normal scores 95%. Abnormal readings are extreme values (std. dev. 0.32 vs 0.15), which a supervised model can exploit; the labelling rule is undocumented.

### 6.4 Discussion

**Effectiveness**

- The project gives a complete, reproducible pipeline from the raw CSV to evaluated models.
- Regression gains consistently over every baseline, though modestly (R² = 0.24, RMSE 13% lower).
- Labelled anomalies are detected reliably by a supervised classifier (F1 = 0.92).

**Limitations**

- About three-quarters of the variance remains unexplained.
- Values are scaled and appear synthetic, so findings may not transfer to real data; Chapter 7 repeats the analysis on a real dataset.
- Only 3.5 months of data are available, so no annual seasonality can be learned.

## 7. Second Dataset and Model Comparison

### 7.1 Dataset

The smart-meter dataset is short and appears synthetic, so the same analysis is repeated on a larger dataset of **power demand and weather**. It records demand every five minutes from 1 January 2021 to 12 December 2024 and contains **393,440 rows**. The values are used in their recorded units; the file does not document the unit of `Power demand`.

**Table 7.1: Power-demand dataset columns.**

| Column | Type | Description |
|---|:---:|:---:|
| datetime | Datetime | Reading time, every 5 minutes (with gaps) |
| Power demand | Numerical | Target: power demand; the unit is not documented |
| temp, dwpt, rhum | Numerical | Air temperature, dew point, relative humidity |
| wdir, wspd, pres | Numerical | Wind direction, wind speed, air pressure |
| moving_avg_3 | Numerical | Precomputed 3-reading average (excluded, see 7.2) |
| year ... minute | Integer | Calendar fields, rebuilt from datetime |

### 7.2 Data Quality and Preprocessing

**Table 7.2: Data quality summary.**

| Check | Result |
|---|:---:|
| Duplicate rows or timestamps | 0 |
| Missing wind direction | 540 rows (0.14%) |
| Expected five-minute slots | 415,009 |
| Observed rows | 393,440 (94.8% coverage) |
| Time gaps | 1,150 gaps; 21,569 missing slots |
| moving_avg_3 | Excluded from the models |

- **Leakage:** `moving_avg_3` equals the mean of the current and the two previous readings, so it already contains the value being predicted. It is excluded.
- **Gaps:** the data are reindexed to a complete five-minute grid, so lags keep their exact timing across gaps; rows without a known target are dropped.
- **Missing values:** missing wind direction is imputed inside each model.
- **Encoding:** wind direction becomes sine and cosine, and hour, weekday and month are encoded cyclically.
- **Calendar fields:** `year`, `month`, `day`, `hour` and `minute` match `datetime` exactly, so they are rebuilt from it.

### 7.3 Demand Patterns

Demand has a clear daily cycle and a strong seasonal cycle. The mean is about 3,180 at 04:00 and about 4,450 at 11:00, and June (4,858) is far above February (3,220). Sunday is the lowest weekday (3,815 against 4,083 on Wednesday), and the yearly mean rises from 3,483 in 2021 to 4,463 in 2024 (to 12 December). Both charts start at zero.

*Figure 7.1: Mean demand by hour of day. (chart in README_Pixel.pdf)*

*Figure 7.2: Mean demand by month. (chart in README_Pixel.pdf)*

### 7.4 Weather and Demand

Unlike the smart-meter data, the weather here is informative. Temperature has the strongest correlation with demand (r = 0.47), followed by dew point (0.29) and wind speed (0.25); pressure (-0.40) and relative humidity (-0.18) are negative. Pressure moves opposite to temperature (r = -0.78), so the two overlap. In the smart-meter data every weather correlation was below 0.01. Correlation does not establish causation.

*Figure 7.3: Correlation of weather variables with demand. (chart in README_Pixel.pdf)*

### 7.5 Forecasting Setup

The target is demand **5 minutes** and **30 minutes** ahead of each forecast origin. The 30-minute horizon matches the reading interval of the smart-meter data. The inputs are the current demand, lags of 5 minutes, 15 minutes, 1 hour and 24 hours, a trailing one-hour mean, the current weather and cyclical calendar features. The split is chronological 80/20 (first 80% for training, last 20% for testing), and `random_state = 42`, as for the smart-meter models.

- **5 minutes ahead:** train 1 January 2021 to 28 February 2024 (313,831 rows); test 28 February to 12 December 2024 (78,458 rows).
- **30 minutes ahead:** train 311,489 rows; test 77,873 rows.
- **Models:** mean baseline, persistence (demand stays at its current value), Linear Regression, Ridge, Random Forest and Gradient Boosting, the same line-up as for the smart-meter data.
- **Target:** the models learn the change from current demand and the current demand is added back, so all errors are in demand units. Tree models cannot predict above the range seen in training, and the test period goes beyond it (peak 8,632 against a training peak of 7,914), so predicting the level makes them look much worse than they are.

```
Skill vs persistence (%) = 100 * (1 - RMSE / RMSE_persistence)
```

Positive skill means the model beats “no change”. R² is near 1 for every model on this dataset, so skill is the useful score.

### 7.6 Forecasting Results

**Table 7.3: Results for demand 5 minutes ahead (test set, 78,458 rows).**

| Model | MAE | RMSE | R² | Skill vs persistence (%) |
|---|:---:|:---:|:---:|:---:|
| Gradient Boosting | 23.1 | 40.7 | 0.999 | 9.3 |
| Linear Regression | 25.8 | 43.2 | 0.999 | 3.7 |
| Ridge | 25.8 | 43.2 | 0.999 | 3.7 |
| Persistence | 28.6 | 44.8 | 0.999 | 0.0 |
| Random Forest | 25.1 | 50.1 | 0.999 | -11.8 |
| Mean baseline | 1269.6 | 1591.0 | -0.331 | -3447.8 |

**Table 7.4: Results for demand 30 minutes ahead (test set, 77,873 rows).**

| Model | MAE | RMSE | R² | Skill vs persistence (%) |
|---|:---:|:---:|:---:|:---:|
| Random Forest | 54.0 | 126.2 | 0.992 | 26.1 |
| Gradient Boosting | 61.5 | 144.7 | 0.989 | 15.2 |
| Linear Regression | 89.2 | 155.8 | 0.987 | 8.7 |
| Ridge | 89.2 | 155.8 | 0.987 | 8.7 |
| Persistence | 114.0 | 170.7 | 0.985 | 0.0 |
| Mean baseline | 1266.0 | 1587.9 | -0.334 | -830.3 |

*Figure 7.4: Skill versus persistence at the two horizons. (chart in README_Pixel.pdf)*

> **Best models:** Gradient Boosting at 5 minutes (9% better than persistence) and Random Forest at 30 minutes (26% better). Longer horizons give the models more room to beat “no change”.

### 7.7 Comparison with the Smart-Meter Dataset

The two datasets use different units and scales (the smart-meter data are min-max scaled to 0-1), so raw MAE and RMSE cannot be compared. R² and skill versus persistence can. Power demand is shown at the 30-minute horizon, which matches the smart-meter interval. For the smart-meter data, persistence is the previous 30-minute reading on the same test split.

**Table 7.5: Smart-meter and power-demand results (30-minute horizon).**

| Model | Smart R² | Smart skill | Power R² | Power skill |
|---|:---:|:---:|:---:|:---:|
| Mean baseline | -0.004 | 29.8% | -0.334 | -830% |
| Persistence | -1.038 | 0 (ref.) | 0.985 | 0 (ref.) |
| Linear Regression | 0.236 | 38.6% | 0.987 | 8.7% |
| Ridge | 0.232 | 38.6% | 0.987 | 8.7% |
| Random Forest | 0.195 | 37.3% | 0.992 | 26.1% |
| Gradient Boosting | 0.217 | 38.2% | 0.989 | 15.2% |

- **Persistence separates the datasets.** In the smart-meter data it is worse than the mean (negative R²) because consecutive readings are almost independent; in the power-demand data it is very hard to beat.
- **R² alone is misleading** when a series is strongly autocorrelated, so skill versus persistence is the fair score.
- **Weather and calendar carry signal** in the power-demand data but not in the smart-meter data.
- **Caveats:** the datasets differ in size, resolution and inputs (the smart-meter data also had `Avg_Past_Consumption`). The comparison shows how the same modelling recipe performs on each dataset; it does not rank the datasets.

### 7.8 Anomaly Screening

The power-demand data have no anomaly labels, so an unsupervised Isolation Forest is used on demand, weather and calendar features, with the contamination set to 1%. It is fitted on 100,000 evenly spaced readings and scored on all 393,440, which flags **3,923 candidate readings (1.00%)**. These are unusual readings for review, not confirmed faults. Without labels, precision and recall cannot be computed (the smart-meter data had labels), so flagged times should be checked against outages, holidays and weather events.

## 8. AI Tools and Ethical Considerations

### 8.1 Use of AI Tools

- Understanding the dataset and its columns.
- Code assistance and debugging.
- Generating analytical summaries.
- Preparing visualizations.
- Report preparation.

### 8.2 Ethical Considerations

- **Data reliability:** Scaled, apparently synthetic data should not be generalised.
- **Privacy:** Real smart-meter data reveals household routines; anonymise and aggregate it.
- **Responsible AI:** Verify AI-generated code and text, and disclose its use.
- **Automation limits:** False alarms and missed events need human review.

## 9. Conclusion and Future Scope

### 9.1 Conclusion

- The dataset has 5,000 half-hourly readings with no missing values or duplicates.
- Only Avg_Past_Consumption relates to consumption (r = 0.32); weather is near zero.
- There is no strong hourly, weekly or monthly seasonality.
- Linear Regression is the best model (R² = 0.24), a modest gain over the baseline.
- Supervised anomaly detection reaches F1 = 0.92; Isolation Forest reaches only 0.40.
- On 393,440 real power-demand readings, temperature matters (r = 0.47), demand has strong daily and seasonal cycles, and models beat persistence by 9% (5 minutes ahead, Gradient Boosting) and 26% (30 minutes ahead, Random Forest).
- Persistence is the right reference: it is very weak on the smart-meter data but very hard to beat on power demand, so skill versus persistence is a fairer score than R².

### 9.2 Future Scope

- Real-world kWh data with holidays and occupancy.
- Advanced models such as LSTM networks.
- A Streamlit dashboard for monitoring.
- IoT energy meters (ESP32 + current sensor), with proper electrical isolation.

## Bibliography

1. Joel Grus, Data Science from Scratch, O'Reilly Media.
2. Wes McKinney, Python for Data Analysis, O'Reilly Media.
3. Gareth James et al., An Introduction to Statistical Learning, Springer.
4. Smart Meter Electricity Consumption Dataset (ziya07), Kaggle. https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset
5. Five-minute power demand and weather dataset, 2021-2024 (supplied CSV file).
