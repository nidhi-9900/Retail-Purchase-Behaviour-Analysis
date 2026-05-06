import pandas as pd
import numpy as np

# load the dataset from the CSV file
main_data = pd.read_csv('Online_Retail_Featured.csv')
main_data['InvoiceDate'] = pd.to_datetime(main_data['InvoiceDate'])

# clean the data by removing missing customers and duplicates
main_data = main_data.dropna(subset=['CustomerID'])
main_data = main_data.drop_duplicates()

# only keep real orders (quantity and price must be more than zero)
main_data = main_data[(main_data['Quantity'] > 0) & (main_data['UnitPrice'] > 0)]
main_data['CustomerID'] = main_data['CustomerID'].astype(int)

if 'TotalPrice' not in main_data.columns: main_data['TotalPrice'] = main_data['Quantity'] * main_data['UnitPrice']
if 'Month' not in main_data.columns: main_data['Month'] = main_data['InvoiceDate'].dt.to_period('M').astype(str)
if 'Hour' not in main_data.columns: main_data['Hour'] = main_data['InvoiceDate'].dt.hour
if 'Day' not in main_data.columns: main_data['Day'] = main_data['InvoiceDate'].dt.day_name()
main_data = main_data.reset_index(drop=True)

fdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
main_data['Day'] = pd.Categorical(main_data['Day'], categories=fdays, ordered=True)

# this function calculates all the numbers and tables needed for the dashboard
def get_metrics(df):
    if len(df) == 0: return None
    
    metrics = {}
    
    # calculate top level numbers
    metrics['total_revenue'] = df['TotalPrice'].sum()
    metrics['total_orders'] = df['InvoiceNo'].nunique()
    metrics['customer_count'] = df['CustomerID'].nunique()
    
    # calculate how much people spend per order (basket size)
    basket = df.groupby('InvoiceNo')['TotalPrice'].sum().reset_index()
    basket.columns = ['InvoiceNo', 'BasketValue']
    basket = basket[basket['BasketValue'] < 1000]
    metrics['basket_data'] = basket
    metrics['avg_order_value'] = round(basket['BasketValue'].mean(), 0) if len(basket)>0 else 0
    
    # find out how many times each customer has bought something
    freq = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    freq.columns = ['CustomerID', 'PurchaseCount']
    metrics['purchase_freq'] = freq
    
    # calculate the total revenue made in each month
    mo = df.groupby('Month')['TotalPrice'].sum().reset_index()
    mo.columns = ['Month', 'Revenue']
    metrics['monthly_revenue'] = mo.sort_values('Month').reset_index(drop=True)
    
    # get the top 10 countries that have the most orders
    top_c = df['Country'].value_counts().head(10).reset_index()
    top_c.columns = ['Country', 'OrderCount']
    top_c['LogOrders'] = np.log1p(top_c['OrderCount']) if not top_c.empty else []
    metrics['top_countries'] = top_c
    
    # find the top 10 best selling products
    top_p = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10).reset_index()
    top_p.columns = ['ProductName', 'UnitsSold']
    top_p['Rank'] = range(1, len(top_p) + 1)
    metrics['top_products'] = top_p
    
    top_cust = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10).reset_index()
    top_cust.columns = ['CustomerID', 'Revenue']
    top_cust['CustomerID'] = top_cust['CustomerID'].astype(str)
    top_cust['Rank'] = range(1, len(top_cust) + 1)
    metrics['top_customers'] = top_cust
    
    # create data for the heatmap to show busy days and hours
    hm = df.groupby(['Day', 'Hour'], observed=False).size().reset_index()
    hm.columns = ['Day', 'Hour', 'Orders']
    if not hm.empty:
        hm_pivot = hm.pivot(index='Day', columns='Hour', values='Orders')
        hm_pivot = hm_pivot.reindex(fdays).fillna(0)
    else:
        hm_pivot = pd.DataFrame()
    metrics['heatmap_pivot'] = hm_pivot
    
    return metrics

# calculate the default numbers when the app first loads
# this prevents the app from crashing before any filters are applied
default_metrics = get_metrics(main_data)
total_revenue = default_metrics['total_revenue']
total_orders = default_metrics['total_orders']
customer_count = default_metrics['customer_count']
avg_order_value = default_metrics['avg_order_value']
basket_data = default_metrics['basket_data']
purchase_freq = default_metrics['purchase_freq']
monthly_revenue = default_metrics['monthly_revenue']
top_countries = default_metrics['top_countries']
top_products = default_metrics['top_products']
top_customers = default_metrics['top_customers']
heatmap_pivot = default_metrics['heatmap_pivot']

if __name__ == '__main__':
    print(f"shape: {main_data.shape[0]} rows {main_data.shape[1]} cols")
