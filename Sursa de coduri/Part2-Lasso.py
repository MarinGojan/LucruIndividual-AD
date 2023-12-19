import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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

# Împărțirea setului de date în setul de antrenare și setul de testare
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Inițializarea modelului Lasso
lasso_model = Lasso(alpha=0.01)

# Antrenarea modelului pe setul de antrenare
lasso_model.fit(X_train, y_train)

# Afișarea coeficienților non-nuli (caracteristici importante)
important_features = X.columns[lasso_model.coef_ != 0]

# Afișarea diagramelor pentru caracteristicile importante
plt.figure(figsize=(10, 6))
plt.bar(important_features, lasso_model.coef_[lasso_model.coef_ != 0])
plt.xlabel('Caracteristici')
plt.ylabel('Coeficient Lasso')
plt.title('Caracteristici Importante identificate prin Lasso')
plt.xticks(rotation=45)
plt.show()
