import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

demand_data = pd.read_csv('demand_data.csv')
weather_data = pd.read_csv('weather_data.csv')
holiday_data = pd.read_csv('holiday_data.csv')

data = pd.merge(demand_data, weather_data, on='date')
data = pd.merge(data, holiday_data, on='date')

data.fillna(method='ffill', inplace=True)
data['holiday'] = data['holiday'].astype(int)

features = data[['temperature', 'humidity', 'wind_speed', 'holiday']]
target = data['electricity_demand']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

peak_times = X_test[X_test['time'].isin(['15:30', '23:00'])]
peak_predictions = model.predict(peak_times)

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Demand', color='blue')
plt.plot(y_test.index, predictions, label='Predicted Demand', color='red')
plt.title('Electricity Demand Prediction')
plt.xlabel('Date')
plt.ylabel('Electricity Demand (MW)')
plt.legend()
plt.show()
