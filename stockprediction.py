import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pandas_ta as ta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\Bourbon\Desktop\StockPrediction\RELIANCE.csv")
df = df.drop(['Symbol', 'Series'], axis=1, errors='ignore')
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Reliance close price.', fontsize=15)
plt.ylabel('Price of reliance stocks.')
df_cleaned= df.dropna(subset=['Trades', 'Deliverable Volume'])
print(df_cleaned.isnull().sum())
df_cleaned.to_csv(r"C:\Users\Bourbon\Desktop\StockPrediction\RELIANCE.csv", index=False)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))


#Observing the two peaks in the graph plot for the overall OHLC data.

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])

#Boxplot for the OHLC data to observe the outliers for the feature extraction.

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()

splitted = df_cleaned['Date'].str.split('-', expand=True)

#Feature Extraction for the OHLC data and generating quarter end data.

df_cleaned['month'] = splitted[1].astype('int')
df_cleaned['year'] = splitted[0].astype('int')
df_cleaned['day'] = splitted[2].astype('int')
df_cleaned['is_quarter_end'] = np.where(df_cleaned['month']%3==0,1,0)
print(df_cleaned.head())

data_grouped = df_cleaned.drop('Date', axis=1).groupby('year').mean()
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()

df_newgroup= df_cleaned.drop('Date', axis=1).groupby('is_quarter_end').mean()
print(df_newgroup.head())
df_newgroup['open-close']  = df_newgroup['Open'] - df_newgroup['Close']
df_newgroup['low-high']  = df_newgroup['Low'] - df_newgroup['High']
# In df_cleaned, create a target for daily price movement
df_cleaned['target'] = np.where(
    df_cleaned['Close'].shift(-1) > df_cleaned['Close'], 1, 0
)
# Drop the last row (NaN in 'target' due to shift)
df_cleaned = df_cleaned[:-1]

df_newgroup = df_cleaned.groupby('is_quarter_end')['target'].mean().reset_index()
plt.figure(figsize=(8, 6))
sb.barplot(
    x='is_quarter_end', 
    y='target', 
    data=df_newgroup, 
    palette='viridis'
)
plt.title('Probability of Price Increase: Quarter-End vs. Non-Quarter-End')
plt.xlabel('Is Quarter-End (1=Yes, 0=No)')
plt.ylabel('Probability of Next-Day Price Increase')
plt.show()
# Check unique values and counts
print("Target Value Counts:")
print(df_cleaned['target'].value_counts())

# Plot the pie chart with dynamic labels
value_counts = df_cleaned['target'].value_counts()
plt.pie(
    value_counts.values,
    labels=value_counts.index,
    autopct='%1.1f%%',
    colors=['#ff9999','#66b3ff']
)
plt.title('Distribution of Target (0=Price ↓, 1=Price ↑)')
plt.show()
plt.figure(figsize=(10, 10)) 

# As our concern is with the highly 
# correlated features only so, we will visualize 
# our heatmap as per that criteria only. 
sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()

##################################################
### MACHINE LEARNING TRAINING MODEL ###
# This code by myself.
# Create 'Open-Close' and 'Low-High' features
df_cleaned['Open-Close'] = df_cleaned['Open'] - df_cleaned['Close']
df_cleaned['Low-High'] = df_cleaned['Low'] - df_cleaned['High']
features = df_cleaned[['Open-Close', 'Low-High', 'is_quarter_end']]
target = df_cleaned['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, Y_valid)
plt.show()
### MACHINE LEARNING MODEL END OF CODE ###
###################################

### Feature Engineering with MACD, RSI, Past 7 day moving average.
# 1. Lagged Features (e.g., 7-day moving average)
df_cleaned['7_day_MA'] = df_cleaned['Close'].rolling(window=7).mean()

# 2. Technical Indicators (RSI and MACD)
# Relative Strength Index (RSI)
df_cleaned['RSI'] = ta.rsi(df_cleaned['Close'], timeperiod=14)

# Moving Average Convergence Divergence (MACD)
df_cleaned['MACD'], df_cleaned['MACD_signal'], df_cleaned['MACD_hist'] = ta.macd(
    df_cleaned['Close'], fastperiod=12, slowperiod=26, signalperiod=9
)

# 3. Volume Changes (Daily percentage change in volume)
df_cleaned['Volume_change'] = df_cleaned['Volume'].pct_change()

# Drop rows with NaN values (created by rolling windows and technical indicators)
df_cleaned = df_cleaned.dropna()

# Display the updated DataFrame with new features
print(df_cleaned.head())

# Save the updated DataFrame to a new CSV file
df_cleaned.to_csv(r"C:\Users\Bourbon\Desktop\StockPrediction\RELIANCE_with_features.csv", index=False)

# Visualize the new features
plt.figure(figsize=(20, 15))

# Plot 7-day Moving Average
plt.subplot(3, 1, 1)
plt.plot(df_cleaned['Close'], label='Close Price')
plt.plot(df_cleaned['7_day_MA'], label='7-Day Moving Average')
plt.title('Close Price vs 7-Day Moving Average')
plt.legend()

# Plot RSI
plt.subplot(3, 1, 2)
plt.plot(df_cleaned['RSI'], label='RSI', color='orange')
plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
plt.title('Relative Strength Index (RSI)')
plt.legend()

# Plot MACD
plt.subplot(3, 1, 3)
plt.plot(df_cleaned['MACD'], label='MACD', color='blue')
plt.plot(df_cleaned['MACD_signal'], label='MACD Signal', color='red')
plt.bar(df_cleaned.index, df_cleaned['MACD_hist'], label='MACD Histogram', color='gray')
plt.title('Moving Average Convergence Divergence (MACD)')
plt.legend()

plt.tight_layout()
plt.show()

# Compare Volume Changes
plt.figure(figsize=(10, 5))
plt.plot(df_cleaned['Volume_change'], label='Volume Change (%)', color='purple')
plt.title('Daily Volume Change (%)')
plt.axhline(0, linestyle='--', color='black', label='Zero Change')
plt.legend()
plt.show()