import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

# Works locally and on Streamlit Cloud
BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "Player_Market_Value_Prediction_Dataset.csv"

# Streamlit layout
st.set_page_config(
    page_title="TransferIQ: Market Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🔍 Player Explorer")
app_mode = st.sidebar.radio("Choose an option", ["🏠 Home", "⚽ Player Explorer", "📊 EDA", "📈 Player Comparison", "📊 Trending Players", "ℹ️ About the App"])

# Home screen content
if app_mode == "🏠 Home":
    st.title("Welcome to **TransferIQ**! ⚽")
    st.markdown("""
    **TransferIQ** is an innovative platform designed to help you explore and predict football players' market values.
    
    🎯 **Key Features**:
    - Explore player profiles 🧐
    - View accurate market value predictions 💵
    - Understand the factors influencing player valuations 🔍
    - Track trending players and their economic impact 📊
    
    🚀 Our platform uses cutting-edge machine learning models to predict and visualize players' future market values based on their performance, injuries, and media presence.

    ### Let's dive into the football world! 👇
    Use the sidebar to explore more features of the app. Enjoy! 🎉
    """)

    # Add a call to action button
    if st.button("🚀 Start Exploring!"):
        st.markdown("You can explore players, compare them, and see their market value predictions! 🎯")

# Player Explorer content
elif app_mode == "⚽ Player Explorer":
    st.title("⚽ TransferIQ: Player Market Value Dashboard")
    st.markdown("Gain insights into player valuation, sentiment, and enriched attributes.")
    
    # Load your dataset
    df = pd.read_csv(data_path)
    
    # Check that the 'Player Name' column exists
    if 'Player Name' in df.columns:
        players_list = df['Player Name'].dropna().unique().tolist()
        selected_player = st.sidebar.selectbox("🎯 Choose a player:", sorted(players_list))
        
        if selected_player:
            player_row = df[df['Player Name'] == selected_player].iloc[0]
            usd_value = player_row['Market Value (M)']
            exchange_rate = 88.73
            inr_value = usd_value * 1_000_000 * exchange_rate

            st.markdown(f"## 📊 Market Value for **{selected_player}**")
            col1, col2 = st.columns([1, 2])
            col1.metric("💵 USD Value", f"${usd_value:.2f} M")
            col2.metric("🇮🇳 INR Value", f"₹{inr_value:,.0f}")
            st.markdown("---")

            st.markdown("### 🧠 Player Profile")
            profile_fields = ['Age', 'Injury Status', 'Sentiment Label', 'Position', 'Sentiment Score (0–1)']  # Adjusted to match your dataset
            profile_cols = st.columns(len(profile_fields))
            for i, field in enumerate(profile_fields):
                if field in player_row:
                    profile_cols[i].write(f"**{field}:** {player_row[field]}")

            st.markdown("---")
            st.markdown("### 📈 Prediction Metrics")
            pred_fields = ['y_test', 'lstm_preds', 'ensemble_preds', 'lstm_market_value', 'ensemble_market_value']
            pred_cols = st.columns(2)
            for i, field in enumerate(pred_fields):
                if field in player_row:
                    pred_cols[i % 2].write(f"**{field}:** {player_row[field]:.4f}")
            st.markdown("---")
        else:
            st.error("The dataset does not contain a 'Player Name' column or it was not properly loaded.")

# Player Comparison content
elif app_mode == "📈 Player Comparison":
    st.title("📈 Compare Players")
    st.markdown("Select multiple players to compare their market value, attributes, and predictions.")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Check if 'Player Name' column exists and proceed
    if 'Player Name' in df.columns:
        players_list = df['Player Name'].dropna().unique().tolist()
        selected_players = st.sidebar.multiselect("🎯 Choose players:", sorted(players_list))
        
        if len(selected_players) > 1:
            comparison_df = df[df['Player Name'].isin(selected_players)]
            
            # Display comparison table
            st.write("### Player Comparison Table")
            st.write(comparison_df[['Player Name', 'Age', 'Position', 'Market Value (M)', 'Sentiment Label']])

            # Plot a comparison of market values
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x="Player Name", y="Market Value (M)", data=comparison_df)
            st.pyplot(fig)
            
            st.markdown("---")
    else:
        st.error("The dataset does not contain a 'Player Name' column.")

# Trending Players content
elif app_mode == "📊 Trending Players":
    st.title("📊 Trending Players")
    st.markdown("Explore the players whose market values are trending the most in recent times.")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Check if 'Player Name' column exists and proceed
    if 'Player Name' in df.columns:
        # Allow users to select how many players to show (Top 100, 50, 20, 10, or 5)
        top_n_options = [100, 50, 20, 10, 5]
        top_n = st.sidebar.selectbox("Choose number of players to display:", top_n_options)
        
        # Sort the dataset based on market value to get the top players
        df_sorted = df.sort_values(by='Market Value (M)', ascending=False)
        top_players = df_sorted.head(top_n)
        
        st.write(f"### Top {top_n} Trending Players")
        st.write(top_players[['Player Name', 'Market Value (M)', 'Sentiment Label']])

        # Plot market value of trending players
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x="Player Name", y="Market Value (M)", data=top_players)
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain a 'Player Name' column.")

# EDA content
elif app_mode == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    
    # File uploader for user to upload CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load the uploaded file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Show the dataset preview
        st.markdown("### Dataset Preview")
        st.write(df.head())
        
        # Display general information about the dataset
        st.markdown("### Data Info")
        st.write(df.info())
        
        # Show summary statistics
        st.markdown("### Summary Statistics")
        st.write(df.describe())
        
        # Visualizations: Correlation heatmap (only for numeric columns)
        st.markdown("### Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr = df[numeric_cols].corr()  # Calculate correlations for numeric columns
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        st.pyplot(plt)

        # Distribution of numeric columns
        st.markdown("### Distribution of Numeric Columns")
        for col in numeric_cols:
            st.write(f"#### {col} Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

        # Sentiment Label distribution (Categorical data)
        st.markdown("### Sentiment Label Distribution")
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Sentiment Label', data=df)
        st.pyplot(plt)

        # Position Count
        st.markdown("### Position Count Plot")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Position', data=df)
        st.pyplot(plt)

        # Display pairplot for selected numeric columns (if there are enough columns)
        st.markdown("### Pairplot of Selected Columns")
        pair_cols = st.multiselect("Select columns for pairplot", options=df.columns.tolist(), default=numeric_cols[:3])
        if len(pair_cols) > 1:
            sns.pairplot(df[pair_cols])
            st.pyplot()

# About the App content
elif app_mode == "ℹ️ About the App":
    st.title("About TransferIQ")
    st.markdown("""
    **TransferIQ** is an advanced analytics platform designed to provide football clubs, analysts, and fans with cutting-edge insights into player valuations. 
    Our system leverages historical data, player performance metrics, and advanced machine learning models to predict a player's market value.
    
    Key Features:
    - **Market Value Predictions:** Get accurate market value predictions based on various factors such as performance, injuries, and sentiment.
    - **Player Sentiment Analysis:** Our app uses sentiment analysis to assess player popularity and media presence, which impacts their market value.
    - **Enriched Player Profiles:** Dive deep into a player's profile to explore their age, injury status, sentiment, and more.
    
    Whether you're a football fan, data enthusiast, or professional scout, TransferIQ gives you a comprehensive tool to understand player economics like never before!
    """)
