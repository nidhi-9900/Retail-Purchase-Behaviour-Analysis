import base64
import io
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import html, dcc, Input, Output, State, callback
import sys
import os

# add parent directory to path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import (main_data, get_metrics, total_revenue, total_orders, customer_count,
                           avg_order_value, basket_data, purchase_freq, monthly_revenue,
                           top_countries, top_products, top_customers, heatmap_pivot)
from src.model import (rfm_data, scaler_obj, kmeans_model, cluster_names,
                       train_silhouette, test_silhouette, test_db_score)

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

def style_fig(fig):
    fig.update_layout(**BASE_LAYOUT)
    return fig

def make_card(title, fig, insight, id=None):
    graph = dcc.Graph(id=id, figure=fig, config={'displayModeBar': False}) if id else dcc.Graph(figure=fig, config={'displayModeBar': False})
    return html.Div([
        html.Div(title, className='chart-title'),
        graph,
        html.Div([
            html.Span('Insight', className='insight-label'),
            insight
        ], className='finding-box')
    ], className='chart-card')

def make_kpi(label, value, cls, id=None):
    return html.Div([
        html.Span(label, className='kpi-label'),
        html.Div(value, className='kpi-number', id=id) if id else html.Div(value, className='kpi-number')
    ], className=f'kpi-card {cls}')


# fig1
fig1 = px.bar(
    top_countries, x='OrderCount', y='Country',
    orientation='h', title='Top 10 Countries by Orders',
    color='LogOrders', color_continuous_scale=AMBER,
    text='OrderCount'
)
fig1.update_traces(
    texttemplate='%{text:,}', textposition='outside',
    textfont=dict(color='#FFFFFF')
)
fig1 = style_fig(fig1)
fig1.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    margin=dict(l=120, t=50, b=50, r=40),
    showlegend=False
)

# fig2
fig2 = px.bar(
    top_products, x='UnitsSold', y='ProductName',
    orientation='h', title='Top 10 Best Selling Products',
    color='Rank', color_continuous_scale=AMBER,
    text='UnitsSold'
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
peak_mo = monthly_revenue.loc[monthly_revenue['Revenue'].idxmax(), 'Month'] if len(monthly_revenue) > 0 else ''

fig3 = px.line(monthly_revenue, x='Month', y='Revenue',
               title='Monthly Revenue Trend')
fig3.update_traces(
    mode='lines+markers+text',
    line=dict(color='#F5A623', width=2.5),
    marker=dict(color='#FFFFFF', size=8, symbol='circle',
                line=dict(color='#121212', width=1.5)),
    text=[f'\u00a3{v/1000:.0f}K' for v in monthly_revenue['Revenue']],
    textposition='top center',
    textfont=dict(size=10, color='#EEEEEE')
)
if peak_mo:
    fig3.add_annotation(
        x=peak_mo, y=monthly_revenue['Revenue'].max(),
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
fig4 = px.histogram(
    purchase_freq, x='PurchaseCount', nbins=50,
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
fig5 = px.histogram(
    basket_data, x='BasketValue', nbins=50,
    title='Basket Size Distribution',
    color_discrete_sequence=['#C0740A']
)
fig5.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
fig5 = style_fig(fig5)

# fig6
fig6 = px.imshow(
    heatmap_pivot, title='Sales Heatmap by Day and Hour',
    color_continuous_scale=[[0, '#1A0A00'], [0.2, '#4A2000'], [0.4, '#8B4500'], [0.7, '#F5A623'], [1.0, '#FFE4A0']],
    aspect='auto',
    zmin=0, zmax=heatmap_pivot.values.max() * 0.85
)
fig6.update_traces(
    xgap=2, ygap=2, hoverongaps=False,
    hovertemplate='%{y}<br>%{x}:00<br>Orders: %{z:,}<extra></extra>'
)
fig6 = style_fig(fig6)
fig6.update_layout(xaxis=dict(dtick=2))

# fig7
fig7 = px.bar(
    top_customers, x='Revenue', y='CustomerID',
    orientation='h', title='Top 10 Customers by Revenue',
    color='Rank', color_continuous_scale=AMBER,
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
import scipy.stats as stats
skew_v = round(stats.skew(basket_data['BasketValue']), 2)
kurt_v = round(stats.kurtosis(basket_data['BasketValue']), 2)

fig8 = px.histogram(
    basket_data, x='BasketValue', nbins=50,
    color_discrete_sequence=['#C0740A'],
    title=f'Revenue Distribution (Skewness: {skew_v}, Kurtosis: {kurt_v})'
)
fig8.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
fig8 = style_fig(fig8)

# fig9
seg_rfm = rfm_data['Segment'].value_counts().reset_index()
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
    textposition='auto',
    insidetextfont=dict(size=11, color='#121212'),
    outsidetextfont=dict(size=11, color='#EEEEEE'),
    marker=dict(line=dict(width=2, color='#0a0a0a'))
)
fig9 = style_fig(fig9)
fig9.update_layout(height=380, uniformtext_minsize=10, margin=dict(t=50, b=60, l=60, r=60))

# fig10
seg_kmeans = rfm_data['KMSegment'].value_counts().reset_index()
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
    textposition='auto',
    insidetextfont=dict(size=11, color='#121212'),
    outsidetextfont=dict(size=11, color='#EEEEEE'),
    pull=pls,
    marker=dict(line=dict(width=2, color='#0a0a0a'))
)
fig10 = style_fig(fig10)
fig10.update_layout(
    height=380,
    uniformtext_minsize=10,
    legend=dict(orientation='v', x=1.05),
    margin=dict(r=120, t=50, b=60, l=60)
)

# ml tbl
cl_summ = rfm_data.groupby('Cluster').agg(
    Recency=('Recency', 'mean'),
    Frequency=('Frequency', 'mean'),
    Monetary=('Monetary', 'mean')
).round(2)

n_rfm = len(rfm_data)
cl_rows = []
for cid, row in cl_summ.iterrows():
    nm = cluster_names.get(cid, f'Cluster {cid}')
    c = (rfm_data['Cluster'] == cid).sum()
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

assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
app = dash.Dash(__name__, suppress_callback_exceptions=True, assets_folder=assets_path)
app.title = 'Retail Dashboard'

# layout
app.layout = html.Div([
    dcc.Store(id='theme-store', data='dark'),
    dcc.Store(id='page-store', data='overview'),

    html.Div([
        html.Div([
            html.Span('', className='nav-brand-accent'),
            html.Span('Retail Purchase Analysis', className='nav-brand-text')
        ], className='nav-brand'),

        html.Div([
            html.Button('Overview', id='btn-overview', className='nav-btn active', n_clicks=0),
            html.Button('Sales', id='btn-sales', className='nav-btn', n_clicks=0),
            html.Button('Customers', id='btn-customers', className='nav-btn', n_clicks=0),
            html.Button('Statistics', id='btn-statistics', className='nav-btn', n_clicks=0),
            html.Button('ML Model', id='btn-mlmodel', className='nav-btn', n_clicks=0),
            html.Button('Upload', id='btn-upload', className='nav-btn', n_clicks=0),
            html.Button('Light Mode', id='theme-toggle', className='theme-btn theme-dark', n_clicks=0),
        ], className='nav-links'),
    ], className='navbar', id='main-navbar'),

    html.Div([
        html.Div([
            html.Label('Country', style={'fontSize':'10px','color':'#a3a3a3',
                       'textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'4px'}),
            dcc.Dropdown(
                id='filter-country',
                options=[{'label': c, 'value': c} for c in sorted(main_data['Country'].unique())],
                multi=True,
                placeholder='All Countries',
                clearable=True,
                style={'backgroundColor':'#1a1a1a','color':'#f8fafc','fontSize':'13px','minWidth':'180px'}
            )
        ]),
        html.Div([
            html.Label('Date Range', style={'fontSize':'10px','color':'#a3a3a3',
                       'textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'4px'}),
            dcc.DatePickerRange(
                id='filter-date',
                start_date=main_data['InvoiceDate'].min(),
                end_date=main_data['InvoiceDate'].max(),
                display_format='YYYY-MM-DD',
                style={'fontSize':'13px'}
            )
        ]),
        html.Div([
            html.Label('Segment', style={'fontSize':'10px','color':'#a3a3a3',
                       'textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'4px'}),
            dcc.Dropdown(
                id='filter-segment',
                options=[{'label': s, 'value': s} for s in ['Regular','Lost','VIP','Loyal']],
                placeholder='All Segments',
                clearable=True,
                style={'backgroundColor':'#1a1a1a','color':'#f8fafc','fontSize':'13px','minWidth':'150px'}
            )
        ]),
        html.Div([
            html.Label('Basket Size', style={'fontSize':'10px','color':'#a3a3a3',
                       'textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'4px'}),
            dcc.Dropdown(
                id='filter-basket',
                options=[{'label': b, 'value': b} for b in ['Small','Medium','Large']],
                placeholder='All Sizes',
                clearable=True,
                style={'backgroundColor':'#1a1a1a','color':'#f8fafc','fontSize':'13px','minWidth':'130px'}
            )
        ]),
    ], style={'display':'flex','gap':'20px','alignItems':'flex-end','padding':'12px 40px',
              'backgroundColor':'#141414','borderBottom':'1px solid #2a2a2a',
              'position':'sticky','top':'60px','zIndex':'9998'}),

    html.Div([
        html.Div([
            html.Div([
                html.H1('Retail Purchase Behaviour Analysis', className='welcome-heading'),
                html.P('Exploratory Data Analysis with Machine Learning', className='welcome-subtitle')
            ], className='welcome-section'),

            html.Div([
                make_kpi('Total Revenue', f'£{total_revenue / 1000:.1f}K', 'kpi-card-1', id='kpi-revenue'),
                make_kpi('Total Orders', f'{total_orders:,}', 'kpi-card-2', id='kpi-orders'),
                make_kpi('Customers', f'{customer_count:,}', 'kpi-card-3', id='kpi-customers'),
                make_kpi('Avg Order', f'£{avg_order_value:.0f}', 'kpi-card-4', id='kpi-avg'),
            ], className='kpi-row'),

            html.Div('Monthly Revenue Trend', className='section-title'),
            html.Div([
                html.Div('Monthly Revenue Trend', className='chart-title'),
                dcc.Graph(id='graph-monthly', figure=fig3, config={'displayModeBar': False}),
                html.Div([
                    html.Span('Insight', className='insight-label'),
                    'November has highest revenue due to Christmas shopping. Stock up all products before October.'
                ], className='finding-box')
            ], className='chart-card'),
        ], id='page-overview', className='page-section'),

        html.Div([
            html.Div([
                html.H2('Sales Analysis'),
                html.P('Country distribution, product performance and peak hours', className='page-subtitle')
            ], className='page-header'),

            html.Div('Orders by Country', className='section-title'),
            make_card('Orders by Country', fig1, 'United Kingdom dominates with 349K+ orders. Next targets for international expansion are Germany and France.', id='graph-country'),
            html.Div('Best Selling Products', className='section-title'),
            make_card('Best Selling Products', fig2, 'Paper Craft Little Birdie is the bestseller with 80,995 units. Always keep high stock of top 3 products.', id='graph-products'),
            html.Div('Sales Heatmap by Day and Hour', className='section-title'),
            make_card('Sales Heatmap by Day and Hour', fig6, 'Tuesday and Thursday 10am to 2pm are peak hours. Schedule promotions and flash sales during these times.', id='graph-heatmap'),
        ], id='page-sales', className='page-section'),

        html.Div([
            html.Div([
                html.H2('Customer Analysis'),
                html.P('Purchase behavior and customer value analysis', className='page-subtitle')
            ], className='page-header'),

            html.Div('Purchase Frequency', className='section-title'),
            make_card('Customer Purchase Frequency', fig4, 'Most customers buy only 1 to 5 times. A loyalty rewards program is needed to increase repeat purchases.', id='graph-freq'),
            html.Div('Basket Size Distribution', className='section-title'),
            make_card('Basket Size Distribution', fig5, 'Average basket size is 200 to 400 GBP. Bundle offers and upselling strategies can increase this value.', id='graph-basket'),
            html.Div('Top 10 Customers by Revenue', className='section-title'),
            make_card('Top 10 Customers by Revenue', fig7, 'Top customer alone generates huge revenue. Give premium membership and personal service to top 10 customers.', id='graph-topcust'),
        ], id='page-customers', className='page-section'),

        html.Div([
            html.Div([
                html.H2('Statistical Analysis'),
                html.P('Revenue distribution and customer segmentation', className='page-subtitle')
            ], className='page-header'),

            html.Div('Revenue Distribution', className='section-title'),
            make_card('Revenue Distribution', fig8, f'Skewness is {skew_v} which means very few very large orders exist. These bulk buyers are the key revenue drivers.', id='graph-revdist'),
            html.Div('Customer Segment Charts', className='section-title'),
            html.Div([
                make_card('Customer Segments (RFM Method)', fig9, '52% customers are Low Value. Target them with special discount offers to increase engagement.', id='graph-rfm'),
                make_card('Customer Segments (K-Means)', fig10, 'Only 0.3% are VIP customers. Create a VIP loyalty program to retain these high value customers.', id='graph-kmeans'),
            ], className='two-col'),
        ], id='page-statistics', className='page-section'),

        html.Div([
            html.Div([
                html.H2('Machine Learning Model'),
                html.P('K-Means clustering for customer segmentation', className='page-subtitle')
            ], className='page-header'),

            html.Div([
                html.Div([
                    html.Div([
                        html.Div('Live Customer Segment Predictor', className='chart-title'),
                        html.P('Enter RFM values manually to predict which segment this customer belongs to using the trained KMeans model.',
                               style={'color': '#a3a3a3', 'fontSize': '13px', 'marginBottom': '20px'}),
                        html.Div([
                            html.Div([
                                html.Label('Recency (days since last purchase)', style={'color': '#a3a3a3', 'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '6px'}),
                                dcc.Input(id='pred-recency', type='number', placeholder='e.g. 30', min=0, style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a', 'color': '#EEEEEE', 'fontSize': '14px'})
                            ], style={'flex': '1'}),
                            html.Div([
                                html.Label('Frequency (number of orders)', style={'color': '#a3a3a3', 'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '6px'}),
                                dcc.Input(id='pred-frequency', type='number', placeholder='e.g. 5', min=0, style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a', 'color': '#EEEEEE', 'fontSize': '14px'})
                            ], style={'flex': '1'}),
                            html.Div([
                                html.Label('Monetary (total spent in £)', style={'color': '#a3a3a3', 'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '6px'}),
                                dcc.Input(id='pred-monetary', type='number', placeholder='e.g. 500', min=0, style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a', 'color': '#EEEEEE', 'fontSize': '14px'})
                            ], style={'flex': '1'}),
                        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px'}),
                        html.Button('Predict Segment', id='predict-btn', n_clicks=0, style={'backgroundColor': '#f59e0b', 'color': '#0a0a0a', 'border': 'none', 'padding': '10px 28px', 'borderRadius': '8px', 'fontWeight': '700', 'cursor': 'pointer', 'fontSize': '14px'}),
                        html.Div(id='predict-result', style={'marginTop': '20px'})
                    ], className='chart-card', style={'margin': '0'}),

                    html.Div([
                        html.Div('Customer Segment Lookup', className='chart-title'),
                        html.P('Enter a Customer ID to see their purchase history and predicted segment', style={'color': '#a3a3a3', 'marginBottom': '16px', 'fontSize': '14px'}),
                        html.Div([
                            dcc.Input(id='cust-input', type='number', placeholder='Enter Customer ID e.g. 14646', style={'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #3a3a3a', 'backgroundColor': '#1a1a1a', 'color': '#EEEEEE', 'fontSize': '14px', 'width': '280px'}),
                            html.Button('Search', id='cust-btn', n_clicks=0, style={'backgroundColor': '#F5A623', 'color': '#0a0a0a', 'border': 'none', 'padding': '10px 24px', 'borderRadius': '8px', 'fontWeight': '600', 'cursor': 'pointer', 'fontSize': '14px'}),
                        ], style={'display': 'flex', 'gap': '12px'}),
                        html.Div(id='cust-result', style={'marginTop': '20px'}),
                    ], className='chart-card', style={'margin': '0'}),
                ], style={'width': '38%', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}),
                
                html.Div([
                    html.Div([
                        html.H3('K-Means Clustering Algorithm'),
                        html.Div([
                            html.Div([html.Div('1', className='step-number'), html.Div([html.H4('RFM Calculation'), html.P(['Recency = Days since last purchase', html.Br(), 'Frequency = Number of orders placed', html.Br(), 'Monetary = Total money spent'])], className='step-content')], className='ml-step'),
                            html.Div([html.Div('2', className='step-number'), html.Div([html.H4('StandardScaler'), html.P(['We scaled the data so all values', html.Br(), 'are in same range.', html.Br(), 'Monetary is large so scaling', html.Br(), 'prevents it from dominating.'])], className='step-content')], className='ml-step'),
                            html.Div([html.Div('3', className='step-number'), html.Div([html.H4('KMeans (4 Clusters)'), html.P(['Algorithm groups similar customers', html.Br(), 'together automatically.', html.Br(), 'We chose 4 clusters because', html.Br(), 'we want 4 customer types.'])], className='step-content')], className='ml-step'),
                            html.Div([html.Div('4', className='step-number'), html.Div([html.H4('Cluster Names'), html.P(['Regular - Average buyers (70%)', html.Br(), 'Lost - Not buying anymore (25%)', html.Br(), 'Loyal - Frequent buyers (5%)', html.Br(), 'VIP - High spenders (0.3%)'])], className='step-content')], className='ml-step'),
                        ], className='ml-steps-grid'),

                        html.H3('Cluster Summary Results', style={'color': '#f59e0b', 'marginTop': '20px', 'marginBottom': '10px'}),
                        html.Table([html.Thead(html.Tr([html.Th('Segment'), html.Th('Customers'), html.Th('Avg Recency (days)'), html.Th('Avg Frequency'), html.Th('Avg Monetary (\u00a3)')])), html.Tbody(cl_rows)], className='results-table'),
                    ], className='ml-box', style={'margin': '0'}),
                ], style={'width': '60%', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}),
            ], style={'display': 'flex', 'gap': '24px', 'alignItems': 'flex-start', 'marginBottom': '20px'}),

            html.Div('Model Testing and Validation', className='section-title'),
            html.Div([
                html.Div([
                    html.Span('Train Silhouette', style={'fontSize':'11px','color':'#a3a3a3','textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'6px'}),
                    html.Span(str(train_silhouette), style={'fontSize':'28px','fontWeight':'700','color':'#f59e0b'}),
                    html.P('target above 0.4', style={'fontSize':'11px','color':'#a3a3a3','margin':'4px 0 0 0'})
                ], style={'padding':'16px','backgroundColor':'#1a1a1a','borderRadius':'8px','borderLeft':'4px solid #f59e0b','flex':'1'}),
                html.Div([
                    html.Span('Test Silhouette', style={'fontSize':'11px','color':'#a3a3a3','textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'6px'}),
                    html.Span(str(test_silhouette), style={'fontSize':'28px','fontWeight':'700','color':'#fbbf24'}),
                    html.P('should be close to train', style={'fontSize':'11px','color':'#a3a3a3','margin':'4px 0 0 0'})
                ], style={'padding':'16px','backgroundColor':'#1a1a1a','borderRadius':'8px','borderLeft':'4px solid #fbbf24','flex':'1'}),
                html.Div([
                    html.Span('Davies-Bouldin Test', style={'fontSize':'11px','color':'#a3a3a3','textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'6px'}),
                    html.Span(str(test_db_score), style={'fontSize':'28px','fontWeight':'700','color':'#d97706'}),
                    html.P('target below 1.0', style={'fontSize':'11px','color':'#a3a3a3','margin':'4px 0 0 0'})
                ], style={'padding':'16px','backgroundColor':'#1a1a1a','borderRadius':'8px','borderLeft':'4px solid #d97706','flex':'1'}),
                html.Div([
                    html.Span('Generalisation', style={'fontSize':'11px','color':'#a3a3a3','textTransform':'uppercase','letterSpacing':'1px','display':'block','marginBottom':'6px'}),
                    html.Span('Good' if abs(train_silhouette - test_silhouette) < 0.05 else 'Check', style={'fontSize':'28px','fontWeight':'700','color':'#f59e0b'}),
                    html.P('train vs test gap', style={'fontSize':'11px','color':'#a3a3a3','margin':'4px 0 0 0'})
                ], style={'padding':'16px','backgroundColor':'#1a1a1a','borderRadius':'8px','borderLeft':'4px solid #92400e','flex':'1'}),
            ], style={'display':'flex','gap':'16px','marginBottom':'20px'}),
            html.Div([
                html.Span('Validation Note', style={'fontSize':'10px','color':'#f59e0b','textTransform':'uppercase','letterSpacing':'2px','display':'block','marginBottom':'4px','fontWeight':'600'}),
                f'Model trained on 80% customers tested on 20%. Train silhouette {train_silhouette} vs test silhouette {test_silhouette}. Davies-Bouldin {test_db_score} on unseen data. {"Small difference confirms model generalises well." if abs(train_silhouette - test_silhouette) < 0.05 else "Check cluster stability."}'
            ], style={'backgroundColor':'rgba(245,158,11,0.06)','padding':'14px 16px','borderRadius':'6px','borderLeft':'3px solid #f59e0b','fontSize':'13px','color':'#f8fafc','lineHeight':'1.6'}),

            html.Div([
                make_card('Customer Segments (RFM Method)', fig9, '52% customers are Low Value. Focus on converting them to Regular buyers first.'),
                make_card('Customer Segments (K-Means)', fig10, 'KMeans confirms customer segments. VIP customers are very rare and need special retention strategy.'),
            ], className='two-col', style={'marginTop': '20px'}),

        ], id='page-mlmodel', className='page-section'),

        html.Div([
            html.H2('Upload Your Data'),
            html.P('Upload any retail CSV or Excel file to get an instant analysis', className='page-subtitle'),
            dcc.Upload(
                id='upload-dataset',
                children=html.Div(['Drag and drop a CSV or Excel file, or click to browse']),
                style={'width':'100%','height':'80px','lineHeight':'80px','borderWidth':'1px',
                       'borderStyle':'dashed','borderRadius':'8px','textAlign':'center',
                       'cursor':'pointer','borderColor':'#f59e0b','color':'#a3a3a3','fontSize':'13px'},
                multiple=False
            ),
            html.Div(id='upload-column-mapper'),
            html.Div(id='upload-output'),
        ], id='page-upload', className='page-section'),

        html.Button('\u2191', className='scroll-top-btn', id='scroll-btn'),

    ], className='main-content'),

    html.Div([
        html.Span('Retail Purchase Behaviour Analysis'),
        html.Span('Data Analysis Dashboard | 2026')
    ], className='dashboard-footer'),

    html.Script('''
        document.getElementById('page-overview').style.display = 'block';
        function showPage(pid) {
            let pgs = document.querySelectorAll('.page-section');
            pgs.forEach(function(p) { p.style.display = 'none'; });
            document.getElementById('page-' + pid).style.display = 'block';
            window.scrollTo({top: 0, behavior: 'smooth'});
        }
        window.onscroll = function() {
            let b = document.querySelector('.scroll-top-btn');
            if (window.pageYOffset > 200) { b.style.display = 'flex'; } else { b.style.display = 'none'; }
        };
        document.querySelector('.scroll-top-btn').onclick = function() { window.scrollTo({top: 0, behavior: 'smooth'}); };
    ''')

], id='main-div', className='app-wrapper dark-mode')

@callback(
    Output('kpi-revenue', 'children'),
    Output('kpi-orders', 'children'),
    Output('kpi-customers', 'children'),
    Output('kpi-avg', 'children'),
    Output('graph-monthly', 'figure'),
    Input('filter-country', 'value'),
    Input('filter-date', 'start_date'),
    Input('filter-date', 'end_date'),
    Input('filter-segment', 'value'),
    Input('filter-basket', 'value'),
)
def update_overview(selected_countries, start_date, end_date, selected_segment, selected_basket):
    filtered = main_data.copy()
    if selected_countries:
        filtered = filtered[filtered['Country'].isin(selected_countries)]
    if start_date and end_date:
        filtered = filtered[(filtered['InvoiceDate'] >= start_date) & (filtered['InvoiceDate'] <= end_date)]
    if selected_segment:
        segment_customers = rfm_data[rfm_data['KMSegment'] == selected_segment]['CustomerID'].tolist()
        filtered = filtered[filtered['CustomerID'].isin(segment_customers)]
    if selected_basket:
        basket_filtered = main_data.groupby('InvoiceNo')['TotalPrice'].sum().reset_index()
        basket_filtered.columns = ['InvoiceNo', 'BasketValue']
        if selected_basket == 'Small':
            valid_invoices = basket_filtered[basket_filtered['BasketValue'] < 20]['InvoiceNo']
        elif selected_basket == 'Medium':
            valid_invoices = basket_filtered[(basket_filtered['BasketValue'] >= 20) & (basket_filtered['BasketValue'] <= 100)]['InvoiceNo']
        else:
            valid_invoices = basket_filtered[basket_filtered['BasketValue'] > 100]['InvoiceNo']
        filtered = filtered[filtered['InvoiceNo'].isin(valid_invoices)]

    rev = f"\u00a3{filtered['TotalPrice'].sum()/1000:.1f}K"
    orders = f"{filtered['InvoiceNo'].nunique():,}"
    customers = f"{filtered['CustomerID'].nunique():,}"
    avg_val = filtered.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    avg = f"\u00a3{avg_val:.0f}" if pd.notnull(avg_val) else "\u00a30"

    monthly_filtered = filtered.groupby('Month')['TotalPrice'].sum().reset_index()
    monthly_filtered.columns = ['Month', 'Revenue']
    monthly_filtered = monthly_filtered.sort_values('Month')

    peak_month = monthly_filtered.loc[monthly_filtered['Revenue'].idxmax(), 'Month'] if len(monthly_filtered) > 0 else ''

    updated_fig = px.line(monthly_filtered, x='Month', y='Revenue', title='Monthly Revenue Trend')
    updated_fig.update_traces(
        mode='lines+markers+text',
        line=dict(color='#F5A623', width=2.5),
        marker=dict(color='#FFFFFF', size=8, symbol='circle', line=dict(color='#121212', width=1.5)),
        text=[f'\u00a3{v/1000:.0f}K' for v in monthly_filtered['Revenue']],
        textposition='top center',
        textfont=dict(size=10, color='#EEEEEE')
    )
    if peak_month:
        updated_fig.add_annotation(
            x=peak_month, y=monthly_filtered['Revenue'].max(),
            text='Peak: Christmas Season', showarrow=True, arrowhead=2,
            bgcolor='#B7590A', font=dict(color='#FFFFFF', size=12)
        )
    updated_fig.update_layout(**BASE_LAYOUT)

    return rev, orders, customers, avg, updated_fig

@callback(
    Output('page-overview', 'style'),
    Output('page-sales', 'style'),
    Output('page-customers', 'style'),
    Output('page-statistics', 'style'),
    Output('page-mlmodel', 'style'),
    Output('page-upload', 'style'),
    Output('btn-overview', 'className'),
    Output('btn-sales', 'className'),
    Output('btn-customers', 'className'),
    Output('btn-statistics', 'className'),
    Output('btn-mlmodel', 'className'),
    Output('btn-upload', 'className'),
    Input('btn-overview', 'n_clicks'),
    Input('btn-sales', 'n_clicks'),
    Input('btn-customers', 'n_clicks'),
    Input('btn-statistics', 'n_clicks'),
    Input('btn-mlmodel', 'n_clicks'),
    Input('btn-upload', 'n_clicks'),
)
def cb_pages(n1, n2, n3, n4, n5, n6):
    trig = dash.callback_context.triggered[0]['prop_id']
    hd = {'display': 'none'}
    sh = {'display': 'block'}
    n = 'nav-btn'
    a = 'nav-btn active'

    if 'btn-sales' in trig:
        return hd, sh, hd, hd, hd, hd, n, a, n, n, n, n
    elif 'btn-customers' in trig:
        return hd, hd, sh, hd, hd, hd, n, n, a, n, n, n
    elif 'btn-statistics' in trig:
        return hd, hd, hd, sh, hd, hd, n, n, n, a, n, n
    elif 'btn-mlmodel' in trig:
        return hd, hd, hd, hd, sh, hd, n, n, n, n, a, n
    elif 'btn-upload' in trig:
        return hd, hd, hd, hd, hd, sh, n, n, n, n, n, a
    else:
        return sh, hd, hd, hd, hd, hd, a, n, n, n, n, n

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
    if not n or not cid: return ''
    row = rfm_data[rfm_data['CustomerID'] == float(cid)]
    if row.empty:
        return html.Div('Customer ID not found in dataset.', style={'color': '#E74C3C', 'padding': '12px', 'backgroundColor': '#1a1a1a', 'borderRadius': '8px'})

    r = row.iloc[0]
    rec = int(r['Recency'])
    freq = int(r['Frequency'])
    mon = round(r['Monetary'], 2)
    seg_rfm_val = r['Segment']
    seg_km = r['KMSegment']

    recs = {
        'High Value': 'This customer is a high spender. Offer exclusive deals and early access to new products.',
        'Loyal': 'This customer orders frequently. Enrol them in a loyalty rewards program.',
        'Recent': 'This customer purchased recently. Send a follow-up offer to convert them into a regular buyer.',
        'Low Value': 'This customer has low engagement. Target with a special discount to increase activity.'
    }
    tip = recs.get(seg_rfm_val, 'Analyse this customer further for targeted marketing.')

    return html.Div([
        html.Div([
            html.Div([html.Span('Customer ID', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(str(int(cid)), style={'fontSize': '20px', 'fontWeight': '700', 'color': '#F5A623'})], style={'flex': '1'}),
            html.Div([html.Span('Recency', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(f'{rec}d', style={'fontSize': '20px', 'fontWeight': '700', 'color': '#F5A623'})], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),
        html.Div([
            html.Div([html.Span('Frequency', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(f'{freq}', style={'fontSize': '20px', 'fontWeight': '700', 'color': '#F5A623'})], style={'flex': '1'}),
            html.Div([html.Span('Monetary', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(f'\u00a3{mon:,.0f}', style={'fontSize': '20px', 'fontWeight': '700', 'color': '#F5A623'})], style={'flex': '1'}),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '16px'}),
        html.Div([html.Span('RFM Segment', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(seg_rfm_val, style={'fontSize': '18px', 'fontWeight': '700', 'color': '#F5A623'})], style={'marginBottom': '16px'}),
        html.Div([html.Span('Business Recommendation', style={'fontSize': '10px', 'color': '#F5A623', 'textTransform': 'uppercase', 'letterSpacing': '2px', 'display': 'block', 'marginBottom': '4px', 'fontWeight': '600'}), tip], style={'backgroundColor': 'rgba(245,158,11,0.06)', 'padding': '14px 16px', 'borderRadius': '6px', 'borderLeft': '3px solid #F5A623', 'fontSize': '13px', 'color': '#EEEEEE', 'lineHeight': '1.6'})
    ])

@callback(
    Output('predict-result', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('pred-recency', 'value'),
    State('pred-frequency', 'value'),
    State('pred-monetary', 'value')
)
def predict_segment(n, recency, frequency, monetary):
    if not n or recency is None or frequency is None or monetary is None: return ''
    input_scaled = scaler_obj.transform([[recency, frequency, monetary]])
    cluster = int(kmeans_model.predict(input_scaled)[0])
    segment = cluster_names.get(cluster, 'Unknown')

    health = 0
    if recency < 30: health += 40
    elif recency < 90: health += 20
    else: health += 5
    if frequency >= 10: health += 30
    elif frequency >= 5: health += 20
    else: health += 5
    if monetary >= 1000: health += 30
    elif monetary >= 300: health += 20
    else: health += 5

    if health >= 70:
        health_color, health_label = '#22c55e', 'Healthy'
    elif health >= 40:
        health_color, health_label = '#f59e0b', 'Average'
    else:
        health_color, health_label = '#ef4444', 'At Risk'

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
        html.Div([html.Span('Predicted Segment', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(segment, style={'fontSize': '24px', 'fontWeight': '700', 'color': '#f59e0b'})], style={'marginBottom': '16px'}),
        html.Div([html.Span('Health Score', style={'fontSize': '11px', 'color': '#a3a3a3', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'display': 'block', 'marginBottom': '4px'}), html.Span(f'{health}/100', style={'fontSize': '24px', 'fontWeight': '700', 'color': health_color}), html.Span(f' {health_label}', style={'fontSize': '14px', 'color': health_color, 'marginLeft': '8px'})], style={'marginBottom': '16px'}),
        html.Div([html.Span('Business Recommendation', style={'fontSize': '10px', 'color': '#f59e0b', 'textTransform': 'uppercase', 'letterSpacing': '2px', 'display': 'block', 'marginBottom': '6px', 'fontWeight': '600'}), tip], style={'backgroundColor': 'rgba(245,158,11,0.06)', 'padding': '14px 16px', 'borderRadius': '6px', 'borderLeft': '3px solid #f59e0b', 'fontSize': '13px', 'color': '#EEEEEE', 'lineHeight': '1.6'})
    ])

@callback(
    Output('upload-column-mapper', 'children'),
    Output('upload-output', 'children'),
    Input('upload-dataset', 'contents'),
    State('upload-dataset', 'filename'),
    prevent_initial_call=True
)
def handle_upload(file_contents, filename):
    if file_contents is None: return '', ''
    content_type, content_string = file_contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            uploaded_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            uploaded_df = pd.read_excel(io.BytesIO(decoded))
    except Exception as error:
        return '', html.Div(f'Error reading file: {error}', style={'color':'#ef4444'})

    col_lower = {col: col.lower() for col in uploaded_df.columns}
    date_col = next((c for c in uploaded_df.columns if any(k in col_lower[c] for k in ['date','invoice','time'])), None)
    customer_col = next((c for c in uploaded_df.columns if any(k in col_lower[c] for k in ['customer','user','client'])), None)
    value_col = next((c for c in uploaded_df.columns if any(k in col_lower[c] for k in ['price','revenue','amount','total','value'])), None)
    country_col = next((c for c in uploaded_df.columns if any(k in col_lower[c] for k in ['country','region','city'])), None)

    summary = html.Div([
        html.P(f'Loaded: {filename} — {uploaded_df.shape[0]:,} rows, {uploaded_df.shape[1]} columns', style={'color':'#f59e0b','fontSize':'13px','marginBottom':'12px'}),
        html.P(f'Detected — Date: {date_col}, Customer: {customer_col}, Value: {value_col}, Geography: {country_col}', style={'color':'#a3a3a3','fontSize':'12px'})
    ])

    charts = []
    if value_col and date_col:
        try:
            uploaded_df[date_col] = pd.to_datetime(uploaded_df[date_col], errors='coerce')
            uploaded_df['upload_month'] = uploaded_df[date_col].dt.to_period('M').astype(str)
            monthly_upload = uploaded_df.groupby('upload_month')[value_col].sum().reset_index()
            upload_trend = px.line(monthly_upload, x='upload_month', y=value_col, title=f'{value_col} Over Time', markers=True)
            style_fig(upload_trend)
            charts.append(make_card('Revenue Over Time', upload_trend, 'Monthly trend from uploaded data.'))
        except: pass

    if country_col and value_col:
        try:
            country_upload = uploaded_df.groupby(country_col)[value_col].sum().nlargest(10).reset_index()
            upload_geo = px.bar(country_upload, x=value_col, y=country_col, orientation='h', title=f'Top 10 by {country_col}', color=value_col, color_continuous_scale=AMBER)
            style_fig(upload_geo)
            charts.append(make_card(f'Top by {country_col}', upload_geo, 'Geographic breakdown from uploaded data.'))
        except: pass

    if not charts:
        charts.append(html.P('Could not detect enough columns for charts. Need at least a date and value column.', style={'color':'#a3a3a3','fontSize':'13px'}))

    return summary, html.Div(charts)

server = app.server

# this is the master callback function that makes the dashboard dynamic
# it listens to the 4 filters at the top and updates all graphs instantly
@callback(
    Output('kpi-revenue', 'children'), Output('kpi-orders', 'children'),
    Output('kpi-customers', 'children'), Output('kpi-avg', 'children'),
    Output('graph-country', 'figure'), Output('graph-products', 'figure'),
    Output('graph-monthly', 'figure'), Output('graph-freq', 'figure'),
    Output('graph-basket', 'figure'), Output('graph-heatmap', 'figure'),
    Output('graph-topcust', 'figure'), Output('graph-revdist', 'figure'),
    Output('graph-rfm', 'figure'), Output('graph-kmeans', 'figure'),
    Input('filter-country', 'value'), Input('filter-date', 'start_date'),
    Input('filter-date', 'end_date'), Input('filter-segment', 'value'),
    Input('filter-basket', 'value')
)
def update_dashboard(countries, start_date, end_date, segment, basket):
    # start with the full dataset
    df = main_data.copy()
    
    # filter by the selected country
    if countries:
        if isinstance(countries, str): countries = [countries]
        df = df[df['Country'].isin(countries)]
    # filter by the selected start and end dates
    if start_date:
        df = df[df['InvoiceDate'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['InvoiceDate'] <= pd.to_datetime(end_date)]
        
    # filter by customer segment using the machine learning results
    if segment:
        from src.model import rfm_data
        seg_custs = rfm_data[rfm_data['Segment'] == segment]['CustomerID']
        df = df[df['CustomerID'].isin(seg_custs)]
        
    # filter by the size of the basket (total price of the order)
    if basket:
        b_df = df.groupby('InvoiceNo')['TotalPrice'].sum().reset_index()
        if basket == 'Small': b_df = b_df[b_df['TotalPrice'] < 50]
        elif basket == 'Medium': b_df = b_df[(b_df['TotalPrice'] >= 50) & (b_df['TotalPrice'] < 200)]
        else: b_df = b_df[b_df['TotalPrice'] >= 200]
        df = df[df['InvoiceNo'].isin(b_df['InvoiceNo'])]

    # Calculate new metrics
    metrics = get_metrics(df)
    
    # If filtered out everything, return empty figures
    if not metrics:
        emp = px.scatter(title='No Data Found')
        emp = style_fig(emp)
        return '£0.0K', '0', '0', '£0', emp, emp, emp, emp, emp, emp, emp, emp, emp, emp

    # KPIs
    kpi_rev = f"£{metrics['total_revenue'] / 1000:.1f}K"
    kpi_ord = f"{metrics['total_orders']:,}"
    kpi_cus = f"{metrics['customer_count']:,}"
    kpi_avg = f"£{metrics['avg_order_value']:.0f}"

    # Figures
    # 1. Country
    f1 = px.bar(metrics['top_countries'], x='OrderCount', y='Country', orientation='h', color='LogOrders', color_continuous_scale=AMBER, text='OrderCount')
    if not metrics['top_countries'].empty: f1.update_traces(texttemplate='%{text:,}', textposition='outside', textfont=dict(color='#FFFFFF'))
    f1 = style_fig(f1).update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=120, t=50, b=50, r=40), showlegend=False)

    # 2. Products
    f2 = px.bar(metrics['top_products'], x='UnitsSold', y='ProductName', orientation='h', color='Rank', color_continuous_scale=AMBER, text='UnitsSold')
    if not metrics['top_products'].empty: f2.update_traces(textposition='auto', cliponaxis=False, textfont=dict(color='#FFFFFF'))
    f2 = style_fig(f2).update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title='Units Sold', showlegend=False)

    # 3. Monthly
    f3 = px.line(metrics['monthly_revenue'], x='Month', y='Revenue')
    if not metrics['monthly_revenue'].empty:
        f3.update_traces(mode='lines+markers+text', line=dict(color='#F5A623', width=2.5), marker=dict(color='#FFFFFF', size=8, line=dict(color='#121212', width=1.5)), text=[f'£{v/1000:.0f}K' for v in metrics['monthly_revenue']['Revenue']], textposition='top center', textfont=dict(size=10, color='#EEEEEE'))
    f3 = style_fig(f3)

    # 4. Freq
    f4 = px.histogram(metrics['purchase_freq'], x='PurchaseCount', nbins=50, color_discrete_sequence=['#C0740A'])
    f4.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
    f4 = style_fig(f4).update_layout(yaxis_title='Number of Customers', bargap=0.05)

    # 5. Basket
    f5 = px.histogram(metrics['basket_data'], x='BasketValue', nbins=50, color_discrete_sequence=['#C0740A'])
    f5.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
    f5 = style_fig(f5)

    # 6. Heatmap
    if not metrics['heatmap_pivot'].empty:
        f6 = px.imshow(metrics['heatmap_pivot'], color_continuous_scale=[[0, '#1A0A00'], [0.2, '#4A2000'], [0.4, '#8B4500'], [0.7, '#F5A623'], [1.0, '#FFE4A0']], aspect='auto', zmin=0, zmax=metrics['heatmap_pivot'].values.max() * 0.85)
        f6.update_traces(xgap=2, ygap=2, hoverongaps=False, hovertemplate='%{y}<br>%{x}:00<br>Orders: %{z:,}<extra></extra>')
    else: f6 = px.scatter(title='No Data')
    f6 = style_fig(f6).update_layout(xaxis=dict(dtick=2))

    # 7. Top Cust
    f7 = px.bar(metrics['top_customers'], x='Revenue', y='CustomerID', orientation='h', color='Rank', color_continuous_scale=AMBER, text='Revenue')
    if not metrics['top_customers'].empty: f7.update_traces(texttemplate='£%{text:,.0f}', textposition='outside', cliponaxis=False, textfont=dict(color='#FFFFFF'))
    f7 = style_fig(f7).update_layout(yaxis={'categoryorder': 'total ascending', 'title': 'Customer ID'}, xaxis=dict(tickprefix='£', tickformat=','), showlegend=False)

    # 8. Rev Dist
    f8 = px.histogram(metrics['basket_data'], x='BasketValue', nbins=50, color_discrete_sequence=['#C0740A'])
    f8.update_traces(marker_line_color='#3D1C00', marker_line_width=0.8)
    f8 = style_fig(f8)

    # 9. RFM Pie
    PIE_COLS = ['#F5A623', '#FFD87A', '#C0740A', '#FFFFFF']
    from src.model import rfm_data
    custs = df['CustomerID'].unique()
    f_rfm = rfm_data[rfm_data['CustomerID'].isin(custs)]
    
    seg_rfm = f_rfm['Segment'].value_counts().reset_index()
    seg_rfm.columns = ['Segment', 'Count']
    f9 = px.pie(seg_rfm, values='Count', names='Segment', color_discrete_sequence=PIE_COLS)
    if not seg_rfm.empty:
        f9.update_traces(textinfo='label+percent', textposition='auto', insidetextfont=dict(size=11, color='#121212'), outsidetextfont=dict(size=11, color='#EEEEEE'), marker=dict(line=dict(width=2, color='#0a0a0a')))
    f9 = style_fig(f9).update_layout(height=380, uniformtext_minsize=10, margin=dict(t=50, b=60, l=60, r=60))

    # 10. KMeans Pie
    seg_kmeans = f_rfm['KMSegment'].value_counts().reset_index()
    seg_kmeans.columns = ['Segment', 'Count']
    idx = seg_kmeans['Count'].idxmin() if not seg_kmeans.empty else None
    pls = [0.15 if i == idx else 0 for i in seg_kmeans.index] if idx is not None else []
    f10 = px.pie(seg_kmeans, values='Count', names='Segment', color_discrete_sequence=PIE_COLS)
    if not seg_kmeans.empty:
        f10.update_traces(textinfo='label+percent', textposition='auto', insidetextfont=dict(size=11, color='#121212'), outsidetextfont=dict(size=11, color='#EEEEEE'), pull=pls, marker=dict(line=dict(width=2, color='#0a0a0a')))
    f10 = style_fig(f10).update_layout(height=380, uniformtext_minsize=10, legend=dict(orientation='v', x=1.05), margin=dict(r=120, t=50, b=60, l=60))

    return kpi_rev, kpi_ord, kpi_cus, kpi_avg, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10

if __name__ == '__main__':
    print('running dash..')
    app.run(debug=False, host='0.0.0.0', port=8055)
