import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse

st.set_page_config(page_title="Laptop Finder AI", layout="wide")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    def clean_price(x):
        try:
            return int(str(x).replace("₹", "").replace(",", "").strip())
        except:
            return np.nan

    df["Price"] = df["Price"].apply(clean_price)
    df = df.dropna(subset=["Price"])
    df["Price"] = df["Price"].astype(int)

    df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").fillna(0).astype(int)
    df["SSD_GB"] = df["SSD"].str.extract(r"(\d+)").fillna(0).astype(int)

    def gpu_flag(x):
        x = str(x)
        return 0 if ("Intel" in x or "UHD" in x or "Iris" in x) else 1

    df["Graphics_Flag"] = df["Graphics"].apply(gpu_flag)
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

    return df

df = load_data()

# ---------------- ML MODEL ----------------
X = df[["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# ---------------- UI ----------------
st.title("💻 Laptop Finder AI")
st.caption("Smart Laptop Recommendation System with Explainable AI")

tabs = st.tabs(["🔍 Recommend", "🔎 Search", "💰 Price Filter", "💡 Use-Case Advisor"])

# =====================================================
# TAB 1: RECOMMEND
# =====================================================
with tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        user_type = st.selectbox("User Type", ["Student", "Office", "Gamer"])
        budget = st.slider("Budget (₹)", 20000, 200000, 60000, step=5000)
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
        ssd = st.selectbox("SSD (GB)", [256, 512, 1024])

    with col2:
        rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0, step=0.1)
        graphics = st.radio("Graphics", ["Integrated", "Dedicated"])

    if st.button("Find Best Laptops"):
        g_flag = 0 if graphics == "Integrated" else 1
        user_input = scaler.transform([[budget, ram, ssd, rating, g_flag]])
        dist, idxs = knn.kneighbors(user_input)

        st.subheader("🎯 Recommended Laptops")
        for i, idx in enumerate(idxs[0], 1):
            row = df.iloc[idx]
            score = max(0, 100 - dist[0][i-1] * 10)
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])

            st.markdown(f"""
            **{i}. {row['Model']}**  
            💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  
            🎮 {row['Graphics']} | ⭐ {row['Rating']}  
            🔍 *Explainable AI:* Similar price, RAM, storage & graphics preference  
            🛒 [Buy on Amazon]({link})
            """)

# =====================================================
# TAB 2: SEARCH
# =====================================================
with tabs[1]:
    query = st.text_input("Search by Brand / Model")

    if query:
        result = df[df["Model"].str.contains(query, case=False, na=False)]

        if result.empty:
            st.warning("No laptops found.")
        else:
            for _, row in result.head(10).iterrows():
                link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
                st.markdown(f"""
                **{row['Model']}**  
                💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  
                🛒 [Amazon]({link})
                """)

# =====================================================
# TAB 3: PRICE FILTER
# =====================================================
with tabs[2]:
    min_p, max_p = st.slider(
        "Select Price Range (₹)",
        int(df["Price"].min()),
        int(df["Price"].max()),
        (30000, 80000),
        step=5000
    )

    filtered = df[(df["Price"] >= min_p) & (df["Price"] <= max_p)]
    filtered = filtered.sort_values(["Rating", "Price"], ascending=[False, True]).head(20)

    st.subheader(f"💰 Laptops between ₹{min_p} – ₹{max_p}")

    for _, row in filtered.iterrows():
        st.markdown(f"""
        **{row['Model']}**  
        💰 ₹{row['Price']} | ⭐ {row['Rating']} | 💾 {row['Ram']}
        """)

# =====================================================
# TAB 4: USE-CASE ADVISOR (EXPLAINABLE AI)
# =====================================================
with tabs[3]:
    use_case = st.selectbox(
        "Select Your Use Case",
        ["Gaming", "Programming", "Video Editing", "Office / Students"]
    )

    if use_case == "Gaming":
        data = df[(df["Graphics_Flag"] == 1) & (df["Ram_GB"] >= 16)]
        reason = "Dedicated GPU + High RAM"
    elif use_case == "Programming":
        data = df[(df["Ram_GB"] >= 8) & (df["SSD_GB"] >= 256)]
        reason = "Fast SSD + sufficient RAM"
    elif use_case == "Video Editing":
        data = df[(df["Graphics_Flag"] == 1) & (df["SSD_GB"] >= 512)]
        reason = "GPU + Large SSD"
    else:
        data = df[df["Rating"] >= 3.5]
        reason = "Good rating & balanced specs"

    st.subheader("💡 Recommended for your use case")
    for _, row in data.sort_values("Rating", ascending=False).head(5).iterrows():
        st.markdown(f"""
        **{row['Model']}**  
        ⭐ {row['Rating']} | 💰 ₹{row['Price']}  
        🔍 *Explainable AI:* {reason}
        """)
