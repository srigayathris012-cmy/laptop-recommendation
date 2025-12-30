import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("💻 Smart Laptop Recommendation System")
st.caption("Find the best laptop for your needs")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

df = load_data()

# Keep original for UI
df_display = df.copy()

# -----------------------------
# Data Cleaning
# -----------------------------
df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(int)
df["Ram_GB"] = df["Ram"].str.extract("(\d+)").astype(int)
df["SSD_GB"] = df["SSD"].str.extract("(\d+)").astype(int)
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

def graphics_flag(x):
    return 0 if "Intel" in str(x) or "UHD" in str(x) else 1

df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)

# -----------------------------
# ML Features
# -----------------------------
X = df[["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn.fit(X_scaled)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🛠 Your Requirements")

budget = st.sidebar.slider("💰 Budget (₹)", 20000, 150000, 60000)
ram = st.sidebar.selectbox("🧠 RAM (GB)", [4, 8, 16, 32])
ssd = st.sidebar.selectbox("💾 SSD (GB)", [256, 512, 1024])
rating = st.sidebar.slider("⭐ Minimum Rating", 40, 100, 60)
graphics = st.sidebar.radio("🎮 Dedicated Graphics?", ["No", "Yes"])

graphics_input = 1 if graphics == "Yes" else 0

# -----------------------------
# Recommendation
# -----------------------------
if st.sidebar.button("🔍 Recommend Laptop"):
    user_input = [[budget, ram, ssd, rating, graphics_input]]
    user_scaled = scaler.transform(user_input)

    distances, indices = knn.kneighbors(user_scaled)

    # Get recommended rows
    rec_df = df_display.iloc[indices[0]]

    # Remove duplicate models
    rec_df = rec_df.drop_duplicates(subset="Model").head(5)

    st.subheader("✅ Recommended Laptops")

    for _, row in rec_df.iterrows():
        with st.container():
            st.markdown(
                f"""
                <div style="border:1px solid #444; padding:15px; border-radius:10px; margin-bottom:15px;">
                    <h3>{row['Model']}</h3>
                    <p><b>Price:</b> {row['Price']}</p>
                    <p><b>RAM:</b> {row['Ram']}</p>
                    <p><b>SSD:</b> {row['SSD']}</p>
                    <p><b>Graphics:</b> {row['Graphics']}</p>
                    <p><b>Display:</b> {row['Display']}</p>
                    <p><b>Rating:</b> ⭐ {row['Rating']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
