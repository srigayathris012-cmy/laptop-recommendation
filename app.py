import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("💻 Laptop Recommendation System")
st.caption("Choose your requirements and get the best laptops")

# ----------------------------------
# Load Dataset
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    df.columns = df.columns.str.strip()
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

df = load_data()
df_display = df.copy()

# ----------------------------------
# Data Cleaning
# ----------------------------------

# Price
df["Price"] = (
    df["Price"]
    .astype(str)
    .str.replace("₹", "")
    .str.replace(",", "")
    .str.extract("(\d+)")
    .fillna(0)
    .astype(int)
)

# RAM
df["Ram_GB"] = df["Ram"].astype(str).str.extract("(\d+)").fillna(0).astype(int)

# SSD
df["SSD_GB"] = (
    df["SSD"]
    .astype(str)
    .str.upper()
    .str.replace("TB", "000")
    .str.replace("GB", "")
    .str.extract("(\d+)")
    .fillna(0)
    .astype(int)
)

# Rating
if "Rating" in df.columns:
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(df["Rating"].mean())
else:
    df["Rating"] = 50

# Graphics Flag
if "Graphics" in df.columns:
    df["Graphics_Flag"] = df["Graphics"].astype(str).apply(
        lambda x: 0 if "INTEL" in x.upper() or "UHD" in x.upper() else 1
    )
else:
    df["Graphics_Flag"] = 0

# ----------------------------------
# ML Model
# ----------------------------------
feature_cols = ["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]
X = df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn.fit(X_scaled)

# ----------------------------------
# Sidebar UI
# ----------------------------------
st.sidebar.header("🛠 Select Your Requirements")

budget_options = {
    "Below ₹40,000": 40000,
    "₹40,000 - ₹60,000": 60000,
    "₹60,000 - ₹80,000": 80000,
    "₹80,000 - ₹1,00,000": 100000,
    "Above ₹1,00,000": 130000
}
budget = st.sidebar.selectbox("💰 Budget", list(budget_options.keys()))
budget_value = budget_options[budget]

ram = st.sidebar.selectbox("🧠 RAM (GB)", [4, 8, 16, 32])
ssd = st.sidebar.selectbox("💾 SSD (GB)", [256, 512, 1024])
rating = st.sidebar.slider("⭐ Minimum Rating", 0, 100, 60)
graphics = st.sidebar.radio("🎮 Dedicated Graphics?", ["No", "Yes"])
graphics_input = 1 if graphics == "Yes" else 0

# ----------------------------------
# Recommendation Button
# ----------------------------------
if st.sidebar.button("🔍 Recommend Laptops"):

    user_input = [[
        budget_value,
        ram,
        ssd,
        rating,
        graphics_input
    ]]

    user_scaled = scaler.transform(user_input)
    distances, indices = knn.kneighbors(user_scaled)

    rec_df = df_display.iloc[indices[0]]
    rec_df = rec_df.drop_duplicates(subset="Model").head(5)

    st.subheader("✅ Recommended Laptops")
        (
            f"""
            <div style="
                background-color:#ffffff;
                color:#000000;
                padding:15px;
                border-radius:10px;
                margin-bottom:15px;
                border:1px solid #ccc;
            ">
                <h3>{row['Model']}</h3>
                <p><b>Price:</b> {row['Price']}</p>
                <p><b>RAM:</b> {row['Ram']}</p>
                <p><b>SSD:</b> {row['SSD']}</p>
                <p><b>Graphics:</b> {row.get('Graphics','N/A')}</p>
                <p><b>Display:</b> {row.get('Display','N/A')}</p>
                <p><b>Rating:</b> ⭐ {row.get('Rating','N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
