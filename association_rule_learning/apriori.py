# Importing the libraries
import pandas as pd
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
goods = []
for i in range(len(dataset)):
    transaction = []
    for item in list(dataset.values[i, :]):
        transaction.append(str(item))
    goods.append(transaction)

# Training the Apriori Model
min_product_transactions_per_day = 3
min_support = min_product_transactions_per_day * 7 / len(dataset)
association_rules = apriori(transactions=goods, min_support=min_support, min_lift=3, max_length=2)
association_results = list(association_rules)

# Adding results to dataframe
df = pd.DataFrame()
for result in association_results:
    items = [item for item in list(result[0])]
    support = result[1]
    confidence = result[2][0][2]
    lift = result[2][0][3]
    df = df.append([[items[0], items[1], support, confidence, lift]])

# Sorting and labelling dataframe
df.columns = ["Item 1", "Item 2", "Support", "Confidence", "Lift"]
df = df.sort_values(by='Lift', ascending=False)
df = df.reset_index(drop=True)
print(df)