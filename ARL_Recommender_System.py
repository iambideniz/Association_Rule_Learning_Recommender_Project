################## ASSOCIATION RULE LEARNING RECOMMENDER ##################

#### Business Problem: recommending products to users at the basket stage.

###### Dataset Story ######

# The Online Retail II dataset includes the sales of a UK-based online store between 01/12/2009 - 09/12/2011.
# This company's product catalog includes souvenirs. They can also be considered promotional items.
# There is also information that most of its customers are wholesalers.

####### NOTE ###########
# The cart information of 3 different users is given below.
# Make the most suitable product suggestion for this basket information.
# Note: Product recommendations can be 1 or more than 1.
# Important note: derive the decision rules from 2010-2011 Germany customers.
# ▪ User 1 product id: 21987
# ▪ User 2 product id: 23235
# ▪ User 3 product id: 22747

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

########################### TASK 1 ####################################

# Select 2010-2011 data and preprocess all data.
# Germany selection will be made in the next step.

df = pd.read_excel("hafta 3/ödev/RFM/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df_copy = df.copy()
df.shape

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.shape


########################### TASK 2 ####################################

##### Generate association rules from Germany customers #####

#### Preparing ARL Data Structure (invoice-product matrix) ####
# Let's make the invoice-product intersections 1 or 0.

df_ger = df[df["Country"] == "Germany"]
check_df(df_ger)
df_ger.shape

# How many of each product were bought?
df_ger.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).head(20)

# Let's pivot the table:
# Let invoices go to rows, description statements to columns. (Instead of description, stock code can also be written)
df_ger.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().iloc[0:20, 0:20]

# Let's write 0 to #NaNs.
df_ger.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).iloc[0:20, 0:20]

# If the product is not received in that invoice, we bring it as 0, if it is received, only 1.
df_ger.groupby(["Invoice", "Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:20, 0:20]

# function:
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_ger)
ger_inv_pro_df = create_invoice_product_df(df_ger, id=True)
ger_inv_pro_df.head()

# bonus: if we want to see which product the stock code is.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

###### Assocation Rules #######

# Let's calculate just #support with apriori:
frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values(by="support", ascending=False).head()

# Let's calculate all other metrics with association_rules:
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values(by="support", ascending=False).head()

# antecedents: previous product code
# consequents: next item code
# antecedent support: stand-alone support of the previous product
# consequent support: stand-alone support of the next product
# support: the support value of both products together
# confidence: the probability of purchasing the second item when the first item is purchased
# lift: how many times the probability of buying the second product increases when the first product is purchased

rules.sort_values(by="lift", ascending=False).head(100)
rules.sort_values(by="confidence", ascending=False).head(100)


########################### TASK 3 ####################################

# What are the names of the products whose IDs are given?
# User 1 product id: 21987
# User 2 product id: 23235
# User 3 product id: 22747

check_id(df_ger, 21987)
check_id(df_ger, 23235)
check_id(df_ger, 22747)

########################### TASK 4 ####################################

# Make a product recommendation for the users in the cart.

# User 1 sample product id: 21987

product_id = 21987
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:2]

# function:
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

arl_recommender(rules, 21987, 1)
#21086

arl_recommender(rules, 23235, 1)
#23244

arl_recommender(rules, 22747, 1)
#22745

########################### TASK 5 ####################################

# What are the names of the products?

check_id(df,21989)
# PACK OF 20 SKULL PAPER NAPKINS

check_id(df,23243)
# SET OF TEA COFFEE SUGAR TINS PANTRY

check_id(df,22745)
# POPPY'S PLAYHOUSE BEDROOM