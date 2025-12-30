import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("💻 Smart Laptop Recommendation System")
st.caption("Get the best laptop based on your requirements")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

df = load_data()
df_display = df.copy()  # keep original for user output

# -----------------------------
# Clean Numeric Columns
# -----------------------------
# Price
df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(int)

# RAM
def parse_ram(x):
    try:
        return int(str(x).split()[0])
    except:
        return 0
df["Ram_GB"] = df["Ram"].apply(parse_ram)

# SSD (handle TB to GB)
def parse_ssd(x):
    try:
        x = str(x).upper()
        if "TB" in x:
            return int(float(x.split()[0]) * 1024)
        elif "GB" in x:
            return int(float(x.split()[0]))
        else:
            return 0
    except:
        return 0

df["SSD_GB"] = df["SSD"].apply(parse_ssd)

# Rating
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

# Graphics
def graphics_flag(x):
    x = str(x)
    if "Intel" in x or "UHD" in x:
        return 0
    return 1

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
# -----
