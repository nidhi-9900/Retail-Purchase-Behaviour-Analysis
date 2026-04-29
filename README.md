# Retail Purchase Behaviour Analysis Dashboard

By: Nidhi Sharma and Ridhi Kumari
Roll No: PST-25-0322 and PST-25-0238
Batch C - 2026

## Live Demo
[Dashboard Link] - hosted on Render

## Project Overview
An interactive retail analytics dashboard built using Python Dash.
Uses real transactional data from UCI Machine Learning Repository.
Analyses customer behaviour, sales trends and segments customers
using RFM analysis and K-Means clustering.

## How to Run Locally
Step 1: pip install -r requirements.txt
Step 2: python dashboard.py
Step 3: Open browser at http://localhost:8051

## Tech Stack
- Python
- Dash + Plotly
- Pandas + NumPy
- Scikit-learn (KMeans)
- SciPy (Statistics)

## Features
- 5 page navigation (Overview, Sales, Customers, Statistics, ML Model)
- Dark and Light mode toggle
- 10 interactive Plotly charts
- Live Customer Segment Predictor (KMeans model)
- Customer ID lookup with RFM analysis
- Health score calculation (0-100)
- Business recommendations per customer
- Scroll to top button

## Pages
1. Overview - KPI cards and monthly revenue trend
2. Sales - Country orders, top products, sales heatmap
3. Customers - Purchase frequency, basket size, top customers
4. Statistics - Revenue distribution, RFM and KMeans segments
5. ML Model - Live predictor + customer lookup + algorithm explanation

## Dataset
Online Retail Dataset - UCI Machine Learning Repository
406,829 transactions from Dec 2010 to Dec 2011
4,338 unique customers across 38 countries

## ML Model
Algorithm: K-Means Clustering (4 clusters)
Features: Recency, Frequency, Monetary (RFM)
Preprocessing: StandardScaler normalization
Segments: Regular, Lost, Loyal, VIP