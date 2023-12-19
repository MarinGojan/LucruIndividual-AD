import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Citirea setului de date
file_path = "C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv"
data = pd.read_csv(file_path)

# Alegeți caracteristicile (X) și variabila țintă (y)
X = data.drop(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type', 'last_review'], axis=1)
y = data['availability_365']

# Convertiți variabilele categorice în variabile dummy
X = pd.get_dummies(X)

# Tratarea valorilor lipsă folosind SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Impărțiți datele în set de antrenare și set de testare
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Inițializați modelul Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Antrenați modelul pe datele de antrenare
rf_model.fit(X_train, y_train)

# Efectuați predicții pe setul de testare
y_pred = rf_model.predict(X_test)

# Calculați acuratețea
accuracy = accuracy_score(y_test, y_pred)
print(f'Acuratețea modelului Random Forest: {accuracy:.2f}')

# Obțineți importanța caracteristicilor
feature_importance = rf_model.feature_importances_

# Creați un DataFrame pentru a afișa importanța fiecărei variabile
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})

# Sortați DataFrame-ul după importanța caracteristicilor
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Afișați importanța fiecărei variabile sub formă de grafic de bare
plt.figure(figsize=(12, 6))
plt.bar(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color='skyblue')
plt.title('Top 10 Caracteristici Importante în Random Forest pentru Disponibilitate (availability_365)')
plt.xlabel('Caracteristici')
plt.ylabel('Importanță')
plt.xticks(rotation=45, ha='right')
plt.show()

# Afișați toate variabilele și importanța lor
print("Importanța fiecărei variabile:")
print(feature_importance_df)
