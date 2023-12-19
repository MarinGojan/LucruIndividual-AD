import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Citirea datelor
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"
data = pd.read_csv(file_path)

# Selecția variabilelor dorite
selected_columns = ['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 'neighbourhood',
                     'latitude', 'longitude', 'room_type', 'price', 'minimum_nights', 'number_of_reviews',
                     'last_review', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

data = data[selected_columns]

# Eliminarea înregistrărilor care conțin valori lipsă în variabila țintă
data = data.dropna(subset=['price'])

# Separarea variabilelor numerice și categorice
numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Imputarea valorilor lipsă pentru variabilele numerice cu mediana
numeric_imputer = SimpleImputer(strategy='median')
data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])

# Eliminarea înregistrărilor care conțin valori lipsă în variabilele categorice
data = data.dropna(subset=categorical_features)

# Selectarea caracteristicilor (X) și variabilei țintă (y) după eliminarea valorilor lipsă
X = data.drop('price', axis=1)
y = data['price']

# Codificarea variabilelor categorice folosind one-hot encoding
X_encoded = pd.get_dummies(X, drop_first=True)

# Împărțirea datelor în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Crearea și antrenarea modelului Random Forest Regressor
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)

# Evaluarea modelului pe setul de test
y_pred = random_forest.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Afișarea rezultatelor
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Vizualizarea importanței caracteristicilor
feature_importances = random_forest.feature_importances_
feature_names = X_encoded.columns

# Crearea unui DataFrame pentru importanța caracteristicilor
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sortarea caracteristicilor după importanță
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Afișarea unui grafic cu bare pentru importanța caracteristicilor
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importanța Caracteristicilor în Random Forest Regression')
plt.xlabel('Importanță')
plt.ylabel('Caracteristică')
plt.show()
