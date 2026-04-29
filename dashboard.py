import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import dash
from dash import html, dcc, Input, Output, State, callback

# load
df = pd.read_csv('Online_Retail_Featured.csv')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"rows: {df.shape[0]} cols: {df.shape[1]}")

# cleaning
df = df.dropna(subset=['CustomerID', 'Description'])
df = df.drop_duplicates()
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

if 'TotalPrice' not in df.columns:
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
if 'Month' not in df.columns:
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)
if 'Hour' not in df.columns:
    df['Hour'] = df['InvoiceDate'].dt.hour
if 'Day' not in df.columns:
    df['Day'] = df['InvoiceDate'].dt.day_name()

df = df.reset_index(drop=True)

# agg
total_rev = df['TotalPrice'].sum()
total_orders = df['InvoiceNo'].nunique()
n_customers = df['CustomerID'].nunique()
basket = df.groupby('InvoiceNo')['TotalPrice'].sum()
avg_order = basket.mean()

BASE_LAYOUT = dict(
    paper_bgcolor='#121212',
    plot_bgcolor='#1A1A1A',
    font=dict(family='Arial', color='#EEEEEE', size=12),
    xaxis=dict(gridcolor='#2A2A2A', linecolor='#444444', zerolinecolor='#2A2A2A'),
    yaxis=dict(gridcolor='#2A2A2A', linecolor='#444444', zerolinecolor='#2A2A2A'),
    coloraxis_colorbar=dict(
        tickfont=dict(color='#EEEEEE'),
        title_font=dict(color='#EEEEEE')
    ),
    margin=dict(t=50, b=50, l=70, r=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#EEEEEE'))
)

AMBER = ['#FFE4A0', '#F5A623', '#C0740A', '#7B3F00', '#3D1C00']

def style_fig(fig, h=420):
    fig.update_layout(**BASE_LAYOUT, height=h)
    return fig

def make_card(title, fig, insight):
    return html.Div([
        html.Div(title, className='chart-title'),
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.Span('Insight', className='insight-label'),
            insight
        ], className='finding-box')
    ], className='chart-card')

def make_kpi(label, value, cls):
    return html.Div([
        html.Span(label, className='kpi-label'),
        html.Div(value, className='kpi-number')
    ], className=f'kpi-card {cls}')


# fig1
top10_co = df['Country'].value_counts().head(10).reset_index()
top10_co.columns = ['Country', 'Orders']
top10_co['log_ord'] = np.log1p(top10_co['Orders'])

fig1 = px.bar(
    top10_co, x='Orders', y='Country',
    orientation='h', title='Top 10 Countries by Orders',
    color='log_ord', color_continuous_scale=AMBER,
    text='Orders'
)
fig1.update_traces(
    texttemplate='%{text:,}', textposition='outside',
    textfont=dict(color='#FFFFFF')
)
fig1 = style_fig(fig1, h=480)
fig1.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    margin=dict(l=120, t=50, b=50, r=40),
    showlegend=False
)

# fig2
top10_prod = df.groupby('Description')['Quantity'].sum()
top10_prod = top10_prod.sort_values(ascending=False).head(10).reset_index()
top10_prod.columns = ['Product', 'Quantity']
top10_prod['rank'] = range(1, 11)

fig2 = px.bar(
    top10_prod, x='Quantity', y='Product',
    orientation='h', title='Top 10 Best Selling Products',
    color='rank', color_continuous_scale=AMBER,
    text='Quantity'
)
fig2.update_traces(
    textposition='auto', cliponaxis=False,
    textfont=dict(color='#FFFFFF')
)
fig2 = style_fig(fig2)
fig2.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    xaxis_title='Units Sold', showlegend=False
)


# fig3
monthly = df.groupby('Month')['TotalPrice'].sum().reset_index()
monthly.columns = ['Month', 'Revenue']
monthly = monthly.sort_values('Month')
peak_mo = monthly.loc[monthly['Revenue'].idxmax(), 'Month']

fig3 = px.line(monthly, x='Month', y='Revenue',
               title='Monthly Revenue Trend')
fig3.update_traces(
    mode='lines+markers+text',
    line=dict(color='#F5A623', width=2.5),
    marker=dict(color='#FFFFFF', size=8, symbol='circle',
                line=dict(color='#121212', width=1.5)),
    text=[f'\u00a3{v/1000:.0f}K' for v in monthly['Revenue']],
    textposition='top center',
    textfont=dict(size=10, color='#EEEEEE')
)
fig3.add_annotation(
    x=peak_mo, y=monthly['Revenue'].max(),
    text='Peak: Christmas Season',
    showarrow=True, arrowhead=2,
    bgcolor='#B7590A', font=dict(color='#FFFFFF', size=12)
)
fig3 = style_fig(fig3)
fig3.update_layout(
    xaxis=dict(gridcolor='#2A2A2A'),
    yaxis=dict(gridcolor='#2A2A2A')
)

# fig4
freq = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
freq.columns = ['CustomerID', 'Purchases']

fig4 = px.histogram(
    freq, x='Purchases', nbins=50,
    title='Customer Purchase Frequency',
    color_discrete_sequence=['#C0740A']
)
fig4.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
fig4 = style_fig(fig4)
fig4.update_layout(
    yaxis_title='Number of Customers',
    bargap=0.05
)

# fig5
bsk_df = basket.reset_index()
bsk_df.columns = ['InvoiceNo', 'BasketSize']
bsk_df = bsk_df[bsk_df['BasketSize'] < 1000]

fig5 = px.histogram(
    bsk_df, x='BasketSize', nbins=50,
    title='Basket Size Distribution',
    color_discrete_sequence=['#C0740A']
)
fig5.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
fig5 = style_fig(fig5)


# fig6
fdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
         'Friday', 'Saturday', 'Sunday']
df['Day'] = pd.Categorical(df['Day'], categories=fdays, ordered=True)

hm = df.groupby(['Day', 'Hour']).size().reset_index(name='Orders')
hm_piv = hm.pivot(index='Day', columns='Hour', values='Orders')
hm_piv = hm_piv.reindex(fdays).fillna(0)

fig6 = px.imshow(
    hm_piv, title='Sales Heatmap by Day and Hour',
    color_continuous_scale=[[0, '#1A0A00'], [0.2, '#4A2000'], [0.4, '#8B4500'], [0.7, '#F5A623'], [1.0, '#FFE4A0']],
    aspect='auto',
    zmin=0, zmax=hm_piv.values.max() * 0.85
)
fig6.update_traces(
    xgap=2, ygap=2, hoverongaps=False,
    hovertemplate='%{y}<br>%{x}:00<br>Orders: %{z:,}<extra></extra>'
)
fig6 = style_fig(fig6)
fig6.update_layout(xaxis=dict(dtick=2))


# fig7
top10_cust = df.groupby('CustomerID')['TotalPrice'].sum()
top10_cust = top10_cust.sort_values(ascending=False).head(10).reset_index()
top10_cust.columns = ['CustomerID', 'Revenue']
top10_cust['CustomerID'] = top10_cust['CustomerID'].astype(str)
top10_cust['rk'] = range(1, 11)

fig7 = px.bar(
    top10_cust, x='Revenue', y='CustomerID',
    orientation='h', title='Top 10 Customers by Revenue',
    color='rk', color_continuous_scale=AMBER,
    text='Revenue'
)
fig7.update_traces(
    texttemplate='\u00a3%{text:,.0f}',
    textposition='outside', cliponaxis=False,
    textfont=dict(color='#FFFFFF')
)
fig7 = style_fig(fig7)
fig7.update_layout(
    yaxis={'categoryorder': 'total ascending', 'title': 'Customer ID'},
    xaxis=dict(tickprefix='\u00a3', tickformat=','),
    showlegend=False
)

# fig8
rev_stats = basket[basket < 1000]
skew_v = round(stats.skew(basket), 2)
kurt_v = round(stats.kurtosis(basket), 2)
rev_df = rev_stats.reset_index()
rev_df.columns = ['InvoiceNo', 'TotalPrice']

fig8 = px.histogram(
    rev_df, x='TotalPrice', nbins=50,
    color_discrete_sequence=['#C0740A'],
    title=f'Revenue Distribution (Skewness: {skew_v}, Kurtosis: {kurt_v})'
)
fig8.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
fig8 = style_fig(fig8)


# fig9
snapshot = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()

rfm['Segment'] = 'Low Value'
rfm.loc[rfm['Monetary'] > rfm['Monetary'].quantile(0.75), 'Segment'] = 'High Value'
rfm.loc[rfm['Frequency'] > rfm['Frequency'].quantile(0.75), 'Segment'] = 'Loyal'
rfm.loc[rfm['Recency'] < 30, 'Segment'] = 'Recent'

seg_rfm = rfm['Segment'].value_counts().reset_index()
seg_rfm.columns = ['Segment', 'Count']

PIE_COLS = ['#F5A623', '#FFD87A', '#C0740A', '#FFFFFF']
pie_txt = ['#FFFFFF', '#121212', '#FFFFFF', '#121212']

fig9 = px.pie(
    seg_rfm, values='Count', names='Segment',
    title='Customer Segments (RFM Method)',
    color_discrete_sequence=PIE_COLS
)
fig9.update_traces(
    textinfo='label+percent',
    textfont=dict(size=12, color=pie_txt),
    marker=dict(line=dict(width=2, color='#0a0a0a'))
)
fig9 = style_fig(fig9)

# fig10
scaler = StandardScaler()
rscaler = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(rscaler)
rfm['Cluster'] = kmeans.labels_

cl_summ = rfm.groupby('Cluster').agg(
    Recency=('Recency', 'mean'),
    Frequency=('Frequency', 'mean'),
    Monetary=('Monetary', 'mean')
).round(2)

cl_names = {0: 'Regular', 1: 'Lost', 2: 'VIP', 3: 'Loyal'}
rfm['KMSegment'] = rfm['Cluster'].map(cl_names)

seg_kmeans = rfm['KMSegment'].value_counts().reset_index()
seg_kmeans.columns = ['Segment', 'Count']

idx = seg_kmeans['Count'].idxmin()
pls = [0.15 if i == idx else 0 for i in seg_kmeans.index]

fig10 = px.pie(
    seg_kmeans, values='Count', names='Segment',
    title='Customer Segments by K-Means',
    color_discrete_sequence=PIE_COLS
)
fig10.update_traces(
    textinfo='label+percent',
    textfont=dict(size=12, color=pie_txt),
    pull=pls,
    marker=dict(line=dict(width=2, color='#0a0a0a'))
)
fig10 = style_fig(fig10)
fig10.update_layout(
    legend=dict(orientation='v', x=1.05),
    margin=dict(r=120, t=50, b=50, l=70)
)
# filter + kpi callbacks are defined after app.layout

# ml tbl
n_rfm = len(rfm)
cl_rows = []
for cid, row in cl_summ.iterrows():
    nm = cl_names.get(cid, f'Cluster {cid}')
    c = (rfm['Cluster'] == cid).sum()
    p = c / n_rfm * 100
    cl_rows.append(
        html.Tr([
            html.Td(nm),
            html.Td(f'{c} ({p:.1f}%)'),
            html.Td(f'{row["Recency"]:.2f}'),
            html.Td(f'{row["Frequency"]:.2f}'),
            html.Td(f'{row["Monetary"]:.2f}')
        ])
    )

# app setup
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Retail Dashboard'

# layout
app.layout = html.Div([
    dcc.Store(id='theme-store', data='dark'),
    dcc.Store(id='page-store', data='overview'),

    html.Div([
        # navbar
        html.Div([
            html.Div([
                html.Span('', className='nav-brand-accent'),
                html.Span('Retail Purchase Analysis',
                          className='nav-brand-text')
            ], className='nav-brand'),

            html.Div([
                html.Button('Overview', id='btn-overview', className='nav-btn active', n_clicks=0),
                html.Button('Sales', id='btn-sales', className='nav-btn', n_clicks=0),
                html.Button('Customers', id='btn-customers', className='nav-btn', n_clicks=0),
                html.Button('Statistics', id='btn-statistics', className='nav-btn', n_clicks=0),
                html.Button('ML Model', id='btn-mlmodel', className='nav-btn', n_clicks=0),
                html.Button('Light Mode', id='theme-toggle', className='theme-btn theme-dark', n_clicks=0),
            ], className='nav-links'),
        ], className='navbar', id='main-navbar'),

        # pages
        html.Div([

            # overview
            html.Div([
                html.Div([
                    html.H1('Retail Purchase Behaviour Analysis',
                            className='welcome-heading'),
                    html.P('Exploratory Data Analysis with Machine Learning',
                           className='welcome-subtitle')
                ], className='welcome-section'),

                html.Div([
                    make_kpi('Total Revenue', f'£{total_rev / 1000:.1f}K', 'kpi-card-1'),
                    make_kpi('Total Orders', f'{total_orders:,}', 'kpi-card-2'),
                    make_kpi('Customers', f'{n_customers:,}', 'kpi-card-3'),
                    make_kpi('Avg Order', f'£{avg_order:.0f}', 'kpi-card-4'),
                ], className='kpi-row'),

                html.Div('Monthly Revenue Trend', className='section-title'),
                make_card(
                    'Monthly Revenue Trend', fig3,
                    'November has highest revenue due to Christmas shopping. Stock up all products before October.'
                ),
            ], id='page-overview', className='page-section'),

            # sales
            html.Div([
                html.Div([
                    html.H2('Sales Analysis'),
                    html.P('Country distribution, product performance and peak hours',
                           className='page-subtitle')
                ], className='page-header'),

                html.Div('Orders by Country', className='section-title'),
                make_card(
                    'Orders by Country', fig1,
                    'United Kingdom dominates with 349K+ orders. Next targets for international expansion are Germany and France.'
                ),
                html.Div('Best Selling Products', className='section-title'),
                make_card(
                    'Best Selling Products', fig2,
                    'Paper Craft Little Birdie is the bestseller with 80,995 units. Always keep high stock of top 3 products.'
                ),
                html.Div('Sales Heatmap by Day and Hour',
                         className='section-title'),
                make_card(
                    'Sales Heatmap by Day and Hour', fig6,
                    'Tuesday and Thursday 10am to 2pm are peak hours. Schedule promotions and flash sales during these times.'
                ),
            ], id='page-sales', className='page-section'),

            # customers
            html.Div([
                html.Div([
                    html.H2('Customer Analysis'),
                    html.P('Purchase behavior and customer value analysis',
                           className='page-subtitle')
                ], className='page-header'),

                html.Div('Purchase Frequency', className='section-title'),
                make_card(
                    'Customer Purchase Frequency', fig4,
                    'Most customers buy only 1 to 5 times. A loyalty rewards program is needed to increase repeat purchases.'
                ),
                html.Div('Basket Size Distribution',
                         className='section-title'),
                make_card(
                    'Basket Size Distribution', fig5,
                    'Average basket size is 200 to 400 GBP. Bundle offers and upselling strategies can increase this value.'
                ),
                html.Div('Top 10 Customers by Revenue',
                         className='section-title'),
                make_card(
                    'Top 10 Customers by Revenue', fig7,
                    'Top customer alone generates huge revenue. Give premium membership and personal service to top 10 customers.'
                ),
            ], id='page-customers', className='page-section'),

            # statistics
            html.Div([
                html.Div([
                    html.H2('Statistical Analysis'),
                    html.P('Revenue distribution and customer segmentation',
                           className='page-subtitle')
                ], className='page-header'),

                html.Div('Revenue Distribution', className='section-title'),
                make_card(
                    'Revenue Distribution', fig8,
                    f'Skewness is {skew_v} which means very few very large orders exist. These bulk buyers are the key revenue drivers.'
                ),
                html.Div('Customer Segment Charts',
                         className='section-title'),
                html.Div([
                    make_card(
                        'Customer Segments (RFM Method)', fig9,
                        '52% customers are Low Value. Target them with special discount offers to increase engagement.'
                    ),
                    make_card(
                        'Customer Segments (K-Means)', fig10,
                        'Only 0.3% are VIP customers. Create a VIP loyalty program to retain these high value customers.'
                    ),
                ], className='two-col'),
            ], id='page-statistics', className='page-section'),

            # ml model
            html.Div([
                html.Div([
                    html.H2('Machine Learning Model'),
                    html.P('K-Means clustering for customer segmentation',
                           className='page-subtitle')
                ], className='page-header'),

                html.Div([
                    html.Div([

                # live segment predictor
                html.Div([
                    html.Div('Live Customer Segment Predictor', className='chart-title'),
                    html.P(
                        'Enter RFM values manually to predict which segment this customer belongs to using the trained KMeans model.',
                        style={'color': '#a3a3a3', 'fontSize': '13px', 'marginBottom': '20px'}
                    ),
                    html.Div([
                        html.Div([
                            html.Label('Recency (days since last purchase)',
                                style={'color': '#a3a3a3', 'fontSize': '11px',
                                       'textTransform': 'uppercase', 'letterSpacing': '1px',
                                       'display': 'block', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='pred-recency', type='number', placeholder='e.g. 30', min=0,
                                style={'width': '100%', 'padding': '10px', 'borderRadius': '8px',
                                       'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a',
                                       'color': '#EEEEEE', 'fontSize': '14px'}
                            )
                        ], style={'flex': '1'}),
                        html.Div([
                            html.Label('Frequency (number of orders)',
                                style={'color': '#a3a3a3', 'fontSize': '11px',
                                       'textTransform': 'uppercase', 'letterSpacing': '1px',
                                       'display': 'block', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='pred-frequency', type='number', placeholder='e.g. 5', min=0,
                                style={'width': '100%', 'padding': '10px', 'borderRadius': '8px',
                                       'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a',
                                       'color': '#EEEEEE', 'fontSize': '14px'}
                            )
                        ], style={'flex': '1'}),
                        html.Div([
                            html.Label('Monetary (total spent in £)',
                                style={'color': '#a3a3a3', 'fontSize': '11px',
                                       'textTransform': 'uppercase', 'letterSpacing': '1px',
                                       'display': 'block', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='pred-monetary', type='number', placeholder='e.g. 500', min=0,
                                style={'width': '100%', 'padding': '10px', 'borderRadius': '8px',
                                       'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a',
                                       'color': '#EEEEEE', 'fontSize': '14px'}
                            )
                        ], style={'flex': '1'}),
                    ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}),
                    html.Button(
                        'Predict Segment', id='predict-btn', n_clicks=0,
                        style={'backgroundColor': '#f59e0b', 'color': '#0a0a0a',
                               'border': 'none', 'padding': '10px 28px',
                               'borderRadius': '8px', 'fontWeight': '700',
                               'cursor': 'pointer', 'fontSize': '14px'}
                    ),
                    html.Div(id='predict-result', style={'marginTop': '20px'})
                ], className='chart-card', style={'margin': '0'}),

                html.Div([
                    html.Div('Customer Segment Lookup', className='chart-title'),
                    html.P('Enter a Customer ID to see their purchase history and predicted segment',
                           style={'color': '#a3a3a3', 'marginBottom': '16px', 'fontSize': '14px'}),
                    html.Div([
                        dcc.Input(
                            id='cust-input', type='number',
                            placeholder='Enter Customer ID e.g. 14646',
                            style={
                                'padding': '10px', 'borderRadius': '8px',
                                'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a',
                                'color': '#EEEEEE', 'fontSize': '14px', 'width': '280px'
                            }
                        ),
                        html.Button(
                            'Search', id='cust-btn', n_clicks=0,
                            style={
                                'backgroundColor': '#F5A623', 'color': '#0a0a0a',
                                'border': 'none', 'padding': '10px 24px',
                                'borderRadius': '8px', 'fontWeight': '600',
                                'cursor': 'pointer', 'fontSize': '14px'
                            }
                        ),
                    ], style={'display': 'flex', 'gap': '12px'}),
                    html.Div(id='cust-result', style={'marginTop': '20px'}),
                ], className='chart-card', style={'margin': '0'}),
                    ], style={'width': '38%', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}),
                    
                    html.Div([
                        html.Div([
                    html.H3('K-Means Clustering Algorithm'),
                    html.Div([
                        html.Div([
                            html.Div('1', className='step-number'),
                            html.Div([
                                html.H4('RFM Calculation'),
                                html.P([
                                    'Recency = Days since last purchase',
                                    html.Br(),
                                    'Frequency = Number of orders placed',
                                    html.Br(),
                                    'Monetary = Total money spent'
                                ])
                            ], className='step-content')
                        ], className='ml-step'),
                        html.Div([
                            html.Div('2', className='step-number'),
                            html.Div([
                                html.H4('StandardScaler'),
                                html.P([
                                    'We scaled the data so all values',
                                    html.Br(),
                                    'are in same range.',
                                    html.Br(),
                                    'Monetary is large so scaling',
                                    html.Br(),
                                    'prevents it from dominating.'
                                ])
                            ], className='step-content')
                        ], className='ml-step'),
                        html.Div([
                            html.Div('3', className='step-number'),
                            html.Div([
                                html.H4('KMeans (4 Clusters)'),
                                html.P([
                                    'Algorithm groups similar customers',
                                    html.Br(),
                                    'together automatically.',
                                    html.Br(),
                                    'We chose 4 clusters because',
                                    html.Br(),
                                    'we want 4 customer types.'
                                ])
                            ], className='step-content')
                        ], className='ml-step'),
                        html.Div([
                            html.Div('4', className='step-number'),
                            html.Div([
                                html.H4('Cluster Names'),
                                html.P([
                                    'Regular - Average buyers (70%)',
                                    html.Br(),
                                    'Lost - Not buying anymore (25%)',
                                    html.Br(),
                                    'Loyal - Frequent buyers (5%)',
                                    html.Br(),
                                    'VIP - High spenders (0.3%)'
                                ])
                            ], className='step-content')
                        ], className='ml-step'),
                    ], className='ml-steps-grid'),

                    html.H3('Cluster Summary Results',
                            style={'color': '#f59e0b',
                                   'marginTop': '20px',
                                   'marginBottom': '10px'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th('Segment'),
                            html.Th('Customers'),
                            html.Th('Avg Recency (days)'),
                            html.Th('Avg Frequency'),
                            html.Th('Avg Monetary (\u00a3)')
                        ])),
                        html.Tbody(cl_rows)
                    ], className='results-table'),
                        ], className='ml-box', style={'margin': '0'}),
                    ], style={'width': '60%', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}),
                ], style={'display': 'flex', 'gap': '24px', 'alignItems': 'flex-start', 'marginBottom': '20px'}),

                html.Div([
                    make_card(
                        'Customer Segments (RFM Method)', fig9,
                        '52% customers are Low Value. Focus on converting them to Regular buyers first.'
                    ),
                    make_card(
                        'Customer Segments (K-Means)', fig10,
                        'KMeans confirms customer segments. VIP customers are very rare and need special retention strategy.'
                    ),
                ], className='two-col'),
            ], id='page-mlmodel', className='page-section'),

            html.Button('\u2191', className='scroll-top-btn',
                        id='scroll-btn'),

        ], className='main-content'),

        html.Div([
            html.Span('Retail Purchase Behaviour Analysis'),
            html.Span('Data Analysis Dashboard | 2026')
        ], className='dashboard-footer'),

        html.Script('''
            document.getElementById('page-overview').style.display = 'block';
            function showPage(pid) {
                let pgs = document.querySelectorAll('.page-section');
                pgs.forEach(function(p) {
                    p.style.display = 'none';
                });
                document.getElementById('page-' + pid).style.display = 'block';
                window.scrollTo({top: 0, behavior: 'smooth'});
            }
            window.onscroll = function() {
                let b = document.querySelector('.scroll-top-btn');
                if (window.pageYOffset > 200) {
                    b.style.display = 'flex';
                } else {
                    b.style.display = 'none';
                }
            };
            document.querySelector('.scroll-top-btn').onclick = function() {
                window.scrollTo({top: 0, behavior: 'smooth'});
            };
        ''')

    ], id='main-div', className='app-wrapper dark-mode'),
])

# cb
@callback(
    Output('page-overview', 'style'),
    Output('page-sales', 'style'),
    Output('page-customers', 'style'),
    Output('page-statistics', 'style'),
    Output('page-mlmodel', 'style'),
    Output('btn-overview', 'className'),
    Output('btn-sales', 'className'),
    Output('btn-customers', 'className'),
    Output('btn-statistics', 'className'),
    Output('btn-mlmodel', 'className'),
    Input('btn-overview', 'n_clicks'),
    Input('btn-sales', 'n_clicks'),
    Input('btn-customers', 'n_clicks'),
    Input('btn-statistics', 'n_clicks'),
    Input('btn-mlmodel', 'n_clicks'),
)
def cb_pages(n1, n2, n3, n4, n5):
    trig = dash.callback_context.triggered[0]['prop_id']
    hd = {'display': 'none'}
    sh = {'display': 'block'}
    n = 'nav-btn'
    a = 'nav-btn active'

    if 'btn-sales' in trig:
        return hd, sh, hd, hd, hd, n, a, n, n, n
    elif 'btn-customers' in trig:
        return hd, hd, sh, hd, hd, n, n, a, n, n
    elif 'btn-statistics' in trig:
        return hd, hd, hd, sh, hd, n, n, n, a, n
    elif 'btn-mlmodel' in trig:
        return hd, hd, hd, hd, sh, n, n, n, n, a
    else:
        return sh, hd, hd, hd, hd, a, n, n, n, n

@callback(
    Output('main-div', 'className'),
    Output('theme-toggle', 'children'),
    Output('theme-toggle', 'className'),
    Input('theme-toggle', 'n_clicks'),
)
def cb_theme(n_clk):
    if n_clk is None:
        n_clk = 0
    if n_clk % 2 == 0:
        return 'app-wrapper dark-mode', 'Light Mode', 'theme-btn theme-dark'
    return 'app-wrapper light-mode', 'Dark Mode', 'theme-btn theme-light'

@callback(
    Output('cust-result', 'children'),
    Input('cust-btn', 'n_clicks'),
    State('cust-input', 'value')
)
def search_customer(n, cid):
    if not n or not cid:
        return ''

    row = rfm[rfm['CustomerID'] == float(cid)]

    if row.empty:
        return html.Div('Customer ID not found in dataset.',
                        style={'color': '#E74C3C', 'padding': '12px',
                               'backgroundColor': '#1a1a1a', 'borderRadius': '8px'})

    r = row.iloc[0]
    rec = int(r['Recency'])
    freq = int(r['Frequency'])
    mon = round(r['Monetary'], 2)
    seg_rfm = r['Segment']
    seg_km = r['KMSegment']

    recs = {
        'High Value': 'This customer is a high spender. Offer exclusive deals and early access to new products.',
        'Loyal': 'This customer orders frequently. Enrol them in a loyalty rewards program.',
        'Recent': 'This customer purchased recently. Send a follow-up offer to convert them into a regular buyer.',
        'Low Value': 'This customer has low engagement. Target with a special discount to increase activity.'
    }
    tip = recs.get(seg_rfm, 'Analyse this customer further for targeted marketing.')

    return html.Div([
        html.Div([
            html.Div([
                html.Span('Customer ID', style={'fontSize': '11px', 'color': '#a3a3a3',
                                                'textTransform': 'uppercase', 'letterSpacing': '1px',
                                                'display': 'block', 'marginBottom': '4px'}),
                html.Span(str(int(cid)), style={'fontSize': '20px', 'fontWeight': '700',
                                                'color': '#F5A623'})
            ], style={'flex': '1'}),
            html.Div([
                html.Span('Recency', style={'fontSize': '11px', 'color': '#a3a3a3',
                                            'textTransform': 'uppercase', 'letterSpacing': '1px',
                                            'display': 'block', 'marginBottom': '4px'}),
                html.Span(f'{rec}d', style={'fontSize': '20px', 'fontWeight': '700',
                                            'color': '#F5A623'})
            ], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),
        
        html.Div([
            html.Div([
                html.Span('Frequency', style={'fontSize': '11px', 'color': '#a3a3a3',
                                              'textTransform': 'uppercase', 'letterSpacing': '1px',
                                              'display': 'block', 'marginBottom': '4px'}),
                html.Span(f'{freq}', style={'fontSize': '20px', 'fontWeight': '700',
                                            'color': '#F5A623'})
            ], style={'flex': '1'}),
            html.Div([
                html.Span('Monetary', style={'fontSize': '11px', 'color': '#a3a3a3',
                                             'textTransform': 'uppercase', 'letterSpacing': '1px',
                                             'display': 'block', 'marginBottom': '4px'}),
                html.Span(f'\u00a3{mon:,.0f}', style={'fontSize': '20px', 'fontWeight': '700',
                                                      'color': '#F5A623'})
            ], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '16px'}),

        html.Div([
            html.Span('RFM Segment', style={'fontSize': '11px', 'color': '#a3a3a3',
                                            'textTransform': 'uppercase', 'letterSpacing': '1px',
                                            'display': 'block', 'marginBottom': '4px'}),
            html.Span(seg_rfm, style={'fontSize': '18px', 'fontWeight': '700', 'color': '#F5A623'})
        ], style={'marginBottom': '16px'}),

        html.Div([
            html.Span('Business Recommendation', style={'fontSize': '10px', 'color': '#F5A623',
                                                        'textTransform': 'uppercase', 'letterSpacing': '2px',
                                                        'display': 'block', 'marginBottom': '4px',
                                                        'fontWeight': '600'}),
            tip
        ], style={'backgroundColor': 'rgba(245,158,11,0.06)', 'padding': '14px 16px',
                  'borderRadius': '6px', 'borderLeft': '3px solid #F5A623',
                  'fontSize': '13px', 'color': '#EEEEEE', 'lineHeight': '1.6'})
    ])


# live ml predictor using trained kmeans + scaler
@callback(
    Output('predict-result', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('pred-recency', 'value'),
    State('pred-frequency', 'value'),
    State('pred-monetary', 'value')
)
def predict_segment(n, recency, frequency, monetary):
    if not n or recency is None or frequency is None or monetary is None:
        return ''

    input_scaled = scaler.transform([[recency, frequency, monetary]])
    cluster = int(kmeans.predict(input_scaled)[0])
    segment = cl_names.get(cluster, 'Unknown')

    health = 0
    if recency < 30:
        health += 40
    elif recency < 90:
        health += 20
    else:
        health += 5
    if frequency >= 10:
        health += 30
    elif frequency >= 5:
        health += 20
    else:
        health += 5
    if monetary >= 1000:
        health += 30
    elif monetary >= 300:
        health += 20
    else:
        health += 5

    if health >= 70:
        health_color = '#22c55e'
        health_label = 'Healthy'
    elif health >= 40:
        health_color = '#f59e0b'
        health_label = 'Average'
    else:
        health_color = '#ef4444'
        health_label = 'At Risk'

    recs = {
        'High Value': 'High spender. Offer exclusive early access deals.',
        'Loyal': 'Frequent buyer. Enrol in loyalty rewards program.',
        'Recent': 'Just purchased. Send follow-up offer to convert to regular.',
        'Low Value': 'Low engagement. Send special discount to re-activate.',
        'Regular': 'Average buyer. Upsell with bundle offers.',
        'Lost': 'Not buying anymore. Send win-back campaign urgently.',
        'VIP': 'Top customer. Assign personal account manager.',
    }
    tip = recs.get(segment, 'Analyse further for targeted marketing.')

    return html.Div([
        html.Div([
            html.Span('Predicted Segment',
                style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase',
                       'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}),
            html.Span(segment,
                style={'fontSize': '24px', 'fontWeight': '700', 'color': '#f59e0b'})
        ], style={'marginBottom': '16px'}),
        
        html.Div([
            html.Span('Health Score',
                style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase',
                       'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}),
            html.Span(f'{health}/100',
                style={'fontSize': '24px', 'fontWeight': '700', 'color': health_color}),
            html.Span(f' {health_label}',
                style={'fontSize': '14px', 'color': health_color, 'marginLeft': '8px'})
        ], style={'marginBottom': '16px'}),
        
        html.Div([
            html.Span('Business Recommendation',
                style={'fontSize': '10px', 'color': '#f59e0b', 'textTransform': 'uppercase',
                       'letterSpacing': '2px', 'display': 'block', 'marginBottom': '6px',
                       'fontWeight': '600'}),
            tip
        ], style={'backgroundColor': 'rgba(245,158,11,0.06)', 'padding': '14px 16px',
                  'borderRadius': '6px', 'borderLeft': '3px solid #f59e0b',
                  'fontSize': '13px', 'color': '#EEEEEE', 'lineHeight': '1.6'})
    ])

server = app.server
if __name__ == '__main__':
    print('running dash..')
    print(f'rows: {df.shape[0]} | cust: {n_customers}')
    
    app.run(debug=False, host='0.0.0.0', port=8051)

