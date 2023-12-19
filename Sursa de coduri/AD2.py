import pandas as pd

# Încărcarea setului de date
csv_data = pd.read_csv("C:/Users/gojan/OneDrive/Desktop/AB_NYC_2019.csv")

# Calcularea medianei pentru coloana 'price'
median_price = csv_data['price'].median()

# Crearea unei noi variabile 'price_category'
csv_data['price_category'] = ['At or Above Median' if x >= median_price else 'Below Median' for x in csv_data['price']]

# Analizarea distribuției categoriilor de preț
price_category_counts = csv_data['price_category'].value_counts()
print(price_category_counts)


