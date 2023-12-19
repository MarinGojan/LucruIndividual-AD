import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# Încărcarea datelor
csv_data = pd.read_csv("C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv")

# Curățarea și preprocesarea datelor
csv_data_cleaned = csv_data.dropna()
csv_data_cleaned = csv_data_cleaned[csv_data_cleaned['price'] < 500]  # Filtrare pentru prețuri extreme

# 1. Harta de căldură a corelațiilor
correlation_matrix = csv_data_cleaned[['price', 'number_of_reviews', 'reviews_per_month', 'availability_365', 'minimum_nights']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlations Among Various Features')
plt.show()

# 2. Regresie liniară
ohe = OneHotEncoder()
transformed_data = ohe.fit_transform(csv_data_cleaned[['neighbourhood_group', 'room_type']])
transformed_df = pd.DataFrame(transformed_data.toarray(), columns=ohe.get_feature_names_out(['neighbourhood_group', 'room_type']))
transformed_df['price'] = csv_data_cleaned['price'].values
X = transformed_df.drop('price', axis=1)
y = transformed_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices: Linear Regression Model')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# 3. Analiza distribuției geografice a listărilor
plt.figure(figsize=(12, 10))
sns.scatterplot(data=csv_data_cleaned, x='longitude', y='latitude', hue='neighbourhood_group', palette='coolwarm', alpha=0.6)
plt.title('Geographical Distribution of Airbnb Listings in New York')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Neighborhood Group')
plt.show()

# 4. Analiza clusterelor
kmeans = KMeans(n_clusters=5, random_state=42)
clustering_features = csv_data_cleaned[['latitude', 'longitude', 'price']]
clustering_features['Cluster'] = kmeans.fit_predict(clustering_features[['latitude', 'longitude', 'price']])
plt.figure(figsize=(12, 10))
sns.scatterplot(data=clustering_features, x='longitude', y='latitude', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('Cluster Analysis of Airbnb Listings in New York')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()

# 5. PCA pentru caracteristicile listărilor
pca_features = csv_data_cleaned[['price', 'number_of_reviews', 'reviews_per_month', 'availability_365', 'minimum_nights']]
scaler = StandardScaler()
pca_features_scaled = scaler.fit_transform(pca_features)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(pca_features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=principal_df, alpha=0.6)
plt.title('PCA of Airbnb Listings in New York')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 1. Boxplot-uri pentru distribuția prețurilor în funcție de diferite cartiere
plt.figure(figsize=(15, 10))
sns.boxplot(data=csv_data_cleaned, x='neighbourhood_group', y='price')
plt.title('Price Distribution in Different Neighborhoods')
plt.xlabel('Neighborhood Group')
plt.ylabel('Price')
plt.ylim(0, 500)  # Limită pentru o vizualizare mai bună
plt.show()

# 2. Matrice de Confuzie și Curba ROC pentru un model de clasificare
# Crearea unei variabile țintă binare pentru clasificarea proprietăților
price_threshold = csv_data_cleaned['price'].median()
csv_data_cleaned['high_price'] = (csv_data_cleaned['price'] > price_threshold).astype(int)

# Pregătirea datelor pentru clasificare
X = csv_data_cleaned[['latitude', 'longitude', 'number_of_reviews', 'reviews_per_month', 'availability_365']]
y = csv_data_cleaned['high_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Antrenarea modelului RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Generarea matricei de confuzie
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Generarea curbei ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 3. Diagramă Tort pentru distribuția tipurilor de camere
room_type_counts = csv_data_cleaned['room_type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(room_type_counts, labels=room_type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Room Types')
plt.show()

plt.figure(figsize=(15, 6))
sns.boxplot(data=csv_data_cleaned, x='neighbourhood_group', y='price')
plt.title('Price Distribution by Neighborhood Group')
plt.ylim(0, 500)  # Limită pentru o vizualizare mai bună
plt.show()

plt.figure(figsize=(15, 6))
sns.boxplot(data=csv_data_cleaned, x='room_type', y='price')
plt.title('Price Distribution by Room Type')
plt.ylim(0, 500)
plt.show()


# Antrenarea modelului DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
y_pred_proba_dt = dt_classifier.predict_proba(X_test)[:, 1]

# Matrice de Confuzie pentru RandomForestClassifier
conf_matrix_rf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d')
plt.title('Confusion Matrix - RandomForest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Matrice de Confuzie pentru DecisionTreeClassifier
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d')
plt.title('Confusion Matrix - DecisionTree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Antrenarea și evaluarea modelului LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lr = lin_reg.predict(X_test)

# Convertim predicțiile de regresie liniară într-un format binar pentru ROC/AUC
y_pred_lr_binary = np.where(y_pred_lr > price_threshold, 1, 0)

# Calcularea curbelor ROC și AUC pentru fiecare model
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_binary)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Vizualizarea curbelor ROC comparate pentru toate cele trei modele
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_dt, tpr_dt, color='green', lw=2, label='Decision Tree (AUC = %0.2f)' % roc_auc_dt)
plt.plot(fpr_lr, tpr_lr, color='red', lw=2, label='Linear Regression (AUC = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(12, 10))
sns.scatterplot(data=csv_data, x='longitude', y='latitude', hue='neighbourhood_group')
plt.title('Distribuția Geografică a Listărilor Airbnb în New York')
plt.legend(title='Grupul de Cartiere')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=csv_data, x='room_type', y='price')
plt.title('Distribuția Prețurilor în Funcție de Tipul de Cameră')
plt.show()


plt.figure(figsize=(12, 6))
sns.lineplot(data=csv_data, x='availability_365', y='number_of_reviews')
plt.title('Relația dintre Disponibilitate și Numărul de Recenzii')
plt.show()




amenities = ['Wi-Fi', 'Air conditioning', 'Kitchen']  # Exemplu de amenajări
counts = [csv_data[csv_data['amenities'].str.contains(amenity)].shape[0] for amenity in amenities]

plt.figure(figsize=(12, 6))
sns.barplot(x=amenities, y=counts)
plt.title('Impactul Amenajărilor asupra Numărului de Listări')
plt.show()

# Presupunând existența unei variabile demografice 'guest_origin'
top_origins = csv_data['guest_origin'].value_counts().head(5)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_origins.index, y=top_origins.values)
plt.title('Top Origini Geografice ale Oaspeților')
plt.show()

# Exemplu de cod pentru vizualizarea frecvenței amenajărilor
# Notă: Acest cod necesită preprocesarea datelor pentru a extrage informații despre amenajări

amenities_count = csv_data['amenities'].value_counts().head(10)  # Top 10 amenajări
plt.figure(figsize=(12, 6))
sns.barplot(x=amenities_count.index, y=amenities_count.values)
plt.title('Top 10 Amenajări în Listările Airbnb')
plt.xticks(rotation=45)
plt.ylabel('Număr de Listări')
plt.xlabel('Amenajări')
plt.show()


# Exemplu de cod pentru vizualizarea originii geografice a oaspeților
# Notă: Acest cod presupune că avem date despre originea oaspeților

guest_origin_count = csv_data['guest_origin'].value_counts().head(5)
plt.figure(figsize=(12, 6))
sns.barplot(x=guest_origin_count.index, y=guest_origin_count.values)
plt.title('Top 5 Origini Geografice ale Oaspeților')
plt.ylabel('Număr de Oaspeți')
plt.xlabel('Origine Geografică')
plt.show()
