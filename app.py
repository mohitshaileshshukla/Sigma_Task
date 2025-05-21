import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE SETUP
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("Customer Segmentation Dashboard")
st.markdown("Explore customer segments by analyzing their demographic and behavioral characteristics.")

try:
    df = pd.read_csv('Customer_Segmentation_Dataset.csv')  
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # DATA CLEANING
    df = df.dropna()

    if 'Dt_Customer' in df.columns:
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

    df['Age'] = 2025 - df['Year_Birth']
    df['Total_Spending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    df['Children'] = df['Kidhome'] + df['Teenhome']

    # Dropping unnecessary columns
    drop_cols = ['ID', 'Year_Birth', 'Dt_Customer']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # One-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    # SIDEBAR
    st.sidebar.header("Clustering Controls")
    k = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=4)

    # DATA SCALING
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)

    # CLUSTERING
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    df['Cluster'] = cluster_labels

    # PCA VISUALIZATION
    st.subheader("Cluster Visualization (PCA)")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60, ax=ax)
    ax.set_title("Customer Segments")
    st.pyplot(fig)

    # CLUSTER PROFILE
    st.subheader("Cluster Profiles (Numeric Averages)")

    # Only numerical columns for calculating mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    cluster_profile = df.groupby('Cluster')[numeric_cols].mean().round(1)
    st.dataframe(cluster_profile.T)

    # CLUSTER SIZE
    st.subheader("Cluster Sizes")
    st.bar_chart(df['Cluster'].value_counts().sort_index())

    # FEATURE DISTRIBUTIONS BY CLUSTER
    st.subheader("Feature Distributions by Cluster")

    cols_to_plot = ['Age', 'Income', 'Total_Spending', 'Children']
    for col in cols_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=df, x='Cluster', y=col, palette='muted', estimator='mean', ax=ax)
        ax.set_title(f"Average {col} per Cluster")
        ax.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal dashed lines
        ax.grid(axis='both')
        st.pyplot(fig)

    # INSIGHTS
    st.subheader("Insights & Business Relevance")
    st.markdown("""
    - **Cluster with highest Total_Spending** may represent VIPs worth prioritizing.
    - **Younger clusters** could respond to online or mobile campaigns.
    - **Families (more 'Children')** may value bundled offers.
    - **Higher income but low spending** clusters may need better targeting.
    """)

    st.markdown("<br><br><p style='text-align: center; font-size: 16px; color: gray;'>Made by Mohit</p>", unsafe_allow_html=True)

except FileNotFoundError:
    st.error("The dataset 'Customer_Segmentation_Dataset.csv' was not found. Please make sure it is in the same directory as the app.")
