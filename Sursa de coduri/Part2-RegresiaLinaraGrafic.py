import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Citirea datelor din fișierul CSV
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"
data = pd.read_csv(file_path)

# Selectarea caracteristicilor relevante și variabilei țintă
selected_features = ['neighbourhood_group', 'room_type', 'number_of_reviews', 'availability_365']
X = data[selected_features]
y = data['price']

# Convertirea caracteristicilor categorice în variabile dummy (one-hot encoding)
X = pd.get_dummies(X)

# Impărțirea datelor în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inițializarea modelului de regresie liniară
model = LinearRegression()

# Antrenarea modelului pe setul de antrenare
model.fit(X_train, y_train)

# Realizarea previziunilor pe setul de testare
y_pred = model.predict(X_test)

# Crearea unui DataFrame pentru a compara actual vs predict
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Afișarea primelor câteva rânduri ale comparației
print(comparison_df.head())

# Crearea unui grafic pentru a compara actual vs predict
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Prices using Linear Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
