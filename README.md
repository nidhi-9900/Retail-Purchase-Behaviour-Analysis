# Retail Purchase Behaviour Analysis Dashboard

## Live Demo
[https://retail-purchase-behaviour-analysis-1.onrender.com/] hosted on Render

## Project Overview
This is a retail data analysis dashboard built using Python. This dashboard helps businesses understand their customers better. It takes raw sales data and turns it into useful charts and numbers. We built this project to show how data science and machine learning can solve real business problems like finding the best customers and knowing when people buy the most.

## Why We Built This
Many retail stores have a lot of data but do not know how to use it. Our dashboard makes it easy to see what is happening in the store. A business owner can use our tool to see total sales, find top selling products, and decide who to send marketing emails to.

## How to Run the Project Locally
If you want to run this project on your own computer, follow these simple steps:

Step 1: Install all required Python packages. Open your terminal and type:
`pip install -r requirements.txt`

Step 2: Start the application by typing:
`python dashboard/app.py`

Step 3: Open your web browser and go to this address:
`http://localhost:8055`

## Technology Used
* **Python**: The main programming language used.
* **Dash and Plotly**: Used to build the web pages and draw the interactive charts.
* **Pandas and NumPy**: Used to clean and organize the large amounts of data.
* **Scikit learn**: Used to build the K Means Machine Learning model.

## Detailed Features and Pages

Our dashboard has a dark mode and light mode option. It also has dynamic filters at the top so you can easily filter the data by Country, Date, Segment, and Basket Size.

Here is what you will find on each page:

### 1. Overview Page
This page shows the main numbers. It displays the total revenue, total orders, and total number of customers. It also has a line chart that shows how the revenue changes month by month.

### 2. Sales Analysis Page
This page focuses on what is selling and where. It shows a bar chart of the countries that place the most orders. It shows the top 10 best selling products. It also features a heatmap that shows the busiest days and hours for sales.

### 3. Customer Analysis Page
This page helps you understand buying habits. It shows how often customers return to buy again. It shows the average basket size (how much people spend per order). It also lists the top 10 customers who spent the most money.

### 4. Statistics Page
This page shows the math behind the sales. It shows the revenue distribution curve. It also contains pie charts that show how many customers belong to each segment based on our machine learning model.

### 5. ML Model Page
This page explains our Machine Learning algorithm. It has two very useful tools:
* **Live Predictor**: You can type in numbers for Recency, Frequency, and Monetary value, and our model will predict which segment that customer belongs to.
* **Customer Lookup**: You can type in a specific Customer ID to instantly see their details, health score, and get a business recommendation on how to handle them.

### 6. Upload Page
This page allows you to test the dashboard with your own data. You can upload a CSV file, and the dashboard will automatically read it and draw charts based on your data.

## Machine Learning Details

We used the **RFM Method** to prepare our data for machine learning. RFM stands for:
* **Recency**: How many days since the customer last bought something.
* **Frequency**: How many separate orders the customer has made.
* **Monetary**: The total amount of money the customer has spent.

After calculating the RFM numbers, we used the **K Means Clustering Algorithm** to group the customers. We tested different numbers of clusters and found that 4 clusters work the best. 

The 4 customer segments are:
* **VIP**: These are the best customers. They spend a lot of money and buy very often.
* **Loyal**: These customers buy often but might spend less per order.
* **Regular**: These are average, normal buyers.
* **Lost**: These customers used to buy from the store but have not bought anything in a long time.

## Dataset Information
We used the Online Retail Dataset from the UCI Machine Learning Repository. It contains 406,829 real transactions from December 2010 to December 2011. There are 4,338 unique customers from 38 different countries in this dataset.
