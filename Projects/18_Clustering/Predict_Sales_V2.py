# import data for offers
import pandas as pd
df_offers = pd.read_excel("./WineKMC.xlsx", sheetname=0)
df_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]

# import data for transactions
df_transactions = pd.read_excel("./WineKMC.xlsx", sheetname=1)
df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['n'] = 1

# estimate total sales per offer_id
df_total_sales = pd.merge(df_offers['offer_id'].to_frame(), df_transactions)
df_total_sales = df_total_sales.groupby(['offer_id']).sum()
df_total_sales.columns = ['Total_Sales']
df_total_sales = df_total_sales.reset_index()

# create final DF (I merging to make sure offer_ids are in order)
df = pd.merge(df_offers,df_total_sales)

# extract y data
y = df[['Total_Sales']]
# onehot encoding for predictors
Xnod = df[df.columns.difference(['Total_Sales','offer_id'])]
X = pd.get_dummies(Xnod, prefix=['cam', 'var','ori'])
X = X.astype(float)

