import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# Load saved files
kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")
product_similarity_df = joblib.load("models/product_similarity.pkl")
rfm = joblib.load("models/rfm_data.pkl")


def recommend_products(product_name, top_n=5):
    product_name = product_name.upper()

    matching_products = [
        product for product in product_similarity_df.index
        if product_name in product.upper()
    ]

    if len(matching_products) == 0:
        return None, []

    selected_product = matching_products[0]

    recommendations = product_similarity_df[selected_product].sort_values(ascending=False)[1:top_n+1]

    return selected_product, recommendations.index.tolist()


def get_segment_label(recency, frequency, monetary):
    if recency <= rfm["Recency"].median() and frequency >= rfm["Frequency"].median() and monetary >= rfm["Monetary"].median():
        return "High-Value Customer"
    elif frequency >= rfm["Frequency"].median() and monetary >= rfm["Monetary"].median():
        return "Regular Customer"
    elif recency > rfm["Recency"].median() and frequency < rfm["Frequency"].median():
        return "At-Risk Customer"
    else:
        return "Occasional Customer"


st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation and Product Recommendation System")

menu = st.sidebar.radio(
    "Select Module",
    ["Product Recommendation", "Customer Segmentation", "Project Overview"]
)


if menu == "Product Recommendation":
    st.header("🎯 Product Recommendation Module")

    st.write("Enter a product name and get 5 similar product recommendations.")

    product_name = st.text_input("Enter Product Name")

    if st.button("Get Recommendations"):
        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        else:
            selected_product, recommendations = recommend_products(product_name)

            if len(recommendations) == 0:
                st.error("Product not found. Try another product name.")
            else:
                st.success(f"Selected Product: {selected_product}")

                st.subheader("Recommended Products")

                for i, product in enumerate(recommendations, start=1):
                    st.write(f"{i}. {product}")


elif menu == "Customer Segmentation":
    st.header("👥 Customer Segmentation Module")

    st.write("Enter Recency, Frequency, and Monetary values to predict customer segment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency in Days", min_value=0, value=30)

    with col2:
        frequency = st.number_input("Frequency", min_value=1, value=5)

    with col3:
        monetary = st.number_input("Monetary Value", min_value=0.0, value=1000.0)

    if st.button("Predict Cluster"):
        input_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        scaled_data = scaler.transform(input_data)

        cluster = kmeans.predict(scaled_data)[0]

        segment = get_segment_label(recency, frequency, monetary)

        st.success(f"Predicted Cluster: {cluster}")
        st.info(f"Customer Segment: {segment}")

        if "High-Value" in segment:
            st.write("This customer is very valuable. Give loyalty benefits and premium offers.")
        elif "Regular" in segment:
            st.write("This customer buys steadily. Send personalized offers.")
        elif "At-Risk" in segment:
            st.write("This customer has not purchased recently. Send discount or retention offers.")
        else:
            st.write("This customer purchases occasionally. Send awareness and seasonal offers.")


else:
    st.header("📌 Project Overview")

    st.write("""
    This project analyzes e-commerce transaction data to:
    
    1. Clean and preprocess transaction records.
    2. Perform Exploratory Data Analysis.
    3. Create RFM features: Recency, Frequency, and Monetary.
    4. Segment customers using KMeans clustering.
    5. Recommend similar products using item-based collaborative filtering.
    6. Deploy the solution using Streamlit.
    """)

    st.subheader("Technologies Used")
    st.write("""
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - KMeans Clustering
    - Cosine Similarity
    - Streamlit
    """)