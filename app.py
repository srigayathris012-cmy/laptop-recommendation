import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import openai
openai.api_key = os.getenv("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")
st.set_page_config(page_title="Laptop Finder AI", layout="wide")

# ================= LOAD & CLEAN DATA =================
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

# ================= ML MODEL =================
X = df[["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# ================= AI ANSWER FUNCTION =================
def ai_answer(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer clearly and politely."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.6,
            max_tokens=300
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return "❌ AI not available. Please check API key or internet."


# ================= SIDEBAR AI =================
st.sidebar.title("🤖 AI Assistant")
user_question = st.sidebar.text_input("Ask anything")

if user_question:
    st.sidebar.markdown("**Answer:**")
    st.sidebar.success(ai_answer(user_question))

# ================= UI =================
st.title("💻 Laptop Finder AI")
st.caption("Laptop Recommendation System using ML + Generative AI")

tabs = st.tabs([
    "🔍 Recommend",
    "🔎 Search",
    "💰 Price Filter",
    "🧠 Smart Insights",
    "📈 Trending"
])

# ================= TAB 1: RECOMMEND =================
with tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
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

        for i, idx in enumerate(idxs[0], 1):
            row = df.iloc[idx]
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
            st.markdown(f"""
            **{i}. {row['Model']}**  
            💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  
            ⭐ {row['Rating']} | 🎮 {row['Graphics']}  
            🛒 [Buy on Amazon]({link})
            """)

# ================= TAB 2: SEARCH =================
with tabs[1]:
    query = st.text_input("Search by Brand or Model")
    if query:
        results = df[df["Model"].str.contains(query, case=False, na=False)]
        if results.empty:
            st.warning("No laptops found")
        else:
            for _, row in results.head(10).iterrows():
                st.write(f"**{row['Model']}** | ₹{row['Price']} | ⭐ {row['Rating']}")

# ================= TAB 3: PRICE FILTER =================
with tabs[2]:
    min_p, max_p = st.slider(
        "Select Price Range",
        int(df.Price.min()),
        int(df.Price.max()),
        (30000, 80000),
        step=5000
    )

    filtered = df[(df.Price >= min_p) & (df.Price <= max_p)]
    for _, row in filtered.head(15).iterrows():
        st.write(f"**{row['Model']}** | ₹{row['Price']} | ⭐ {row['Rating']}")

# ================= TAB 4: SMART INSIGHTS =================
with tabs[3]:
    laptop = st.selectbox("Select Laptop", df["Model"].unique())
    row = df[df["Model"] == laptop].iloc[0]

    st.metric("Price", f"₹{row['Price']}")
    st.metric("Rating", row["Rating"])
    st.metric("RAM", row["Ram"])
    st.metric("SSD", row["SSD"])

# ================= TAB 5: TRENDING =================
with tabs[4]:
    df["Trending_Score"] = (
        (df["Rating"] * 20) +
        (df["Ram_GB"] * 2) +
        (df["SSD_GB"] / 256 * 10)
    )

    trending = df.sort_values("Trending_Score", ascending=False).head(10)

    for _, row in trending.iterrows():
        st.write(f"🔥 **{row['Model']}** | ₹{row['Price']} | ⭐ {row['Rating']}")
