import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse
import os
import google.generativeai as genai
from PIL import Image

# ================= PAGE CONFIG =================
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

# ================= GEMINI AI SETUP =================
genai.configure(api_key=os.getenv("AIzaSyAswyTKSinwbffaYQM3YE9VeBOdcSK0iRo"))
try:
    gemini_model = genai.GenerativeModel("gemini-pro")
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ================= AI ANSWER FUNCTION =================
def ai_answer(question):
    q = question.lower()
    # Dataset-based answers (fast fallback)
    if "under" in q and "k" in q:
        try:
            budget = int("".join(filter(str.isdigit, q))) * 1000
            results = df[df["Price"] <= budget].sort_values(["Rating", "Price"], ascending=[False, True]).head(3)
            if results.empty:
                return f"No laptops found under ₹{budget}"
            text = f"💻 Best laptops under ₹{budget}:\n"
            for _, r in results.iterrows():
                text += f"- {r['Model']} (₹{r['Price']}, ⭐ {r['Rating']})\n"
            return text
        except:
            pass
    if "gaming" in q:
        best = df[df["Graphics_Flag"] == 1].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"🎮 Best gaming laptop: {best['Model']} | ₹{best['Price']} | ⭐ {best['Rating']}"
    if "student" in q:
        best = df[df["Price"] <= 60000].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"📚 Best student laptop: {best['Model']} | ₹{best['Price']}"

    # Gemini AI answers for anything else
    if GEMINI_AVAILABLE:
        try:
            response = gemini_model.generate_content(
                f"""
                You are a Laptop Recommendation AI.
                Answer simply and clearly.

                Question: {question}
                """
            )
            return response.text
        except:
            return "❌ Gemini AI error. Please try again."

    return "⚠️ AI not available. Please check GEMINI_API_KEY."

# ================= GEMINI CHAT UI =================
st.sidebar.title("🤖 Ask Gemini")
try:
    gemini_logo = Image.open("gemini_logo.png")  # Add your Gemini logo in project folder
    st.sidebar.image(gemini_logo, width=80)
except:
    pass

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.sidebar.text_input("Ask Gemini:", key="input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    reply = ai_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.sidebar.chat_message("user").markdown(msg["content"])
    else:
        st.sidebar.chat_message("assistant").markdown(msg["content"])

# ================= MAIN UI =================
st.title("💻 Laptop Finder AI")
st.caption("Smart Laptop Recommendation System with Gemini AI")

tabs = st.tabs([
    "🔍 Recommend",
    "🔎 Search",
    "💰 Price Filter",
    "🧠 Smart Laptop Insights",
    "📈 Trending Laptops"
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
        g_flag = 0 if graphics=="Integrated" else 1
        user_input = scaler.transform([[budget, ram, ssd, rating, g_flag]])
        dist, idxs = knn.kneighbors(user_input)
        for i, idx in enumerate(idxs[0],1):
            row = df.iloc[idx]
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
            st.markdown(f"""
            **{i}. {row['Model']}**  
            💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  
            🎮 {row['Graphics']} | ⭐ {row['Rating']}  
            🛒 [Buy on Amazon]({link})
            """)

# ================= TAB 2: SEARCH =================
with tabs[1]:
    query = st.text_input("Search Laptop (Brand / Model)")
    if query:
        results = df[df["Model"].str.contains(query, case=False, na=False)]
        if results.empty:
            st.warning("No laptops found.")
        else:
            st.dataframe(results.head(10))

# ================= TAB 3: PRICE FILTER =================
with tabs[2]:
    min_p, max_p = st.slider(
        "Select Price Range (₹)",
        int(df["Price"].min()),
        int(df["Price"].max()),
        (30000, 80000),
        step=5000
    )
    filtered = df[(df["Price"] >= min_p) & (df["Price"] <= max_p)]
    st.dataframe(filtered.sort_values("Rating", ascending=False).head(20))

# ================= TAB 4: SMART INSIGHTS =================
with tabs[3]:
    laptop_name = st.selectbox("Select a Laptop", df["Model"].unique())
    row = df[df["Model"]==laptop_name].iloc[0]
    st.subheader("Why this laptop?")
    st.write(f"💰 Price: ₹{row['Price']}")
    st.write(f"💾 RAM: {row['Ram']}")
    st.write(f"💿 SSD: {row['SSD']}")
    st.write(f"⭐ Rating: {row['Rating']}")

# ================= TAB 5: TRENDING =================
with tabs[4]:
    df["Trending_Score"] = (df["Rating"]*20) + (df["Ram_GB"]*5) + (df["SSD_GB"]/10)
    trending = df.sort_values("Trending_Score", ascending=False).head(10)
    st.dataframe(trending[["Model","Price","Rating"]])
