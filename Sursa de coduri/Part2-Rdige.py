import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Citirea setului de date
data = pd.read_csv("C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv")

# Eliminarea coloanelor nedorite sau non-numerice
data = data.select_dtypes(include=[float, int])

# Separarea setului de date în variabila de caracteristici (X) și variabila de rezultat (y)
X = data.drop('price', axis=1)  # Înlocuiește 'Rating' cu 'price'
y = data['price']

# Tratarea valorilor lipsă folosind SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

ridge_model = Ridge(alpha=10.0)  # Ajustează valoarea alpha
ridge_model.fit(X_imputed, y)
correlation_matrix = X_imputed.corr()


# Afișarea coeficienților importanți
coefficients = pd.Series(ridge_model.coef_, index=X.columns)

# Selectarea caracteristicilor cu coeficienți cu magnitudine mare
significant_features = coefficients[coefficients.abs() > 100].index

# Afișarea diagramelor pentru coeficienții importanți
plt.figure(figsize=(12, 6))
coefficients[significant_features].plot(kind='bar')
plt.title("Coeficienți Importanți pentru selecția caracteristicilor folosind Ridge")
plt.ylabel("Coeficient")
plt.xlabel("Caracteristică")
plt.show()
