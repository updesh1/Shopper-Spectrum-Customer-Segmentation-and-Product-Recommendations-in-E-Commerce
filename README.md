# Shopper Spectrum: Customer Segmentation and Product Recommendations

## Project Overview

This project analyzes e-commerce transaction data to segment customers and recommend similar products.

The project uses RFM analysis for customer segmentation and item-based collaborative filtering for product recommendation.

## Features

- Data cleaning
- Exploratory Data Analysis
- RFM analysis
- KMeans customer segmentation
- Product recommendation using cosine similarity
- Streamlit web app

## Dataset Columns

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

## Customer Segmentation

The project uses:

- Recency
- Frequency
- Monetary

Customers are segmented into:

- High-Value
- Regular
- Occasional
- At-Risk

## Product Recommendation

The system recommends 5 similar products based on customer purchase behavior.

## How to Run

### 1. Clone Repository

```bash
git clone <your-repository-link>
cd Shopper_Spectrum