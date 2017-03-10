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

# create final DF
df_offers_and_sales = pd.merge(df_offers,df_total_sales)

# encode categorical predictors
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df = df_offers_and_sales.iloc[:,1:]
categorical_columns = ['campaign', 'varietal','origin', 'past_peak']
for col in categorical_columns:
    # get values
    cat = df[col].values;
    # learn categories
    le.fit(cat)
    # encode into float and allocate
    df[col] = le.transform(cat)

# extract data 
y = df.Total_Sales
X = df.iloc[:,0:5]