import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Sample historical data
data = pd.Series([10, 12, 15, 18, 20, 22, 25, 28, 30, 32], index=pd.date_range('2020-01-01', periods=10, freq='Y'))

# Plot historical data
plt.figure(figsize=(10,6))
plt.plot(data, label='Historical Data')
plt.title('Heart Disease Cases Over Time')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()

# Fit ARIMA model
model = ARIMA(data, order=(1,1,1))
model_fit = model.fit()

# Forecast future cases
forecast, stderr, conf_int = model_fit.forecast(steps=5)

# Plot forecast
plt.figure(figsize=(10,6))
plt.plot(data, label='Historical Data')
plt.plot(np.arange(len(data), len(data)+5), forecast, label='Forecast', linestyle='--', marker='o')
plt.title('Heart Disease Cases Forecast')
plt.xlabel('Year')
plt.ylabel('Cases')
plt.legend()
plt.show()
