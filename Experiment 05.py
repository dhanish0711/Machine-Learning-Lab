# Apply Linear Regression algorithm on dataset to evaluate prediction accuracy.

dfa=pd.read_csv('Iris.csv')
dfa

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dfa = dfa.dropna()

X_iris = dfa[['PetalLengthCm']]
y_iris = dfa['PetalWidthCm']

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train_iris, y_train_iris)

y_pred_iris = linear_model.predict(X_test_iris)

mse = mean_squared_error(y_test_iris, y_pred_iris)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_iris, y_pred_iris)
r2 = r2_score(y_test_iris, y_pred_iris)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")
