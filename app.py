import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse
import re
import openai  # Add OpenAI

st.set_page_config(page_title="Laptop Finder AI", layout="wide")

# ---------------- LOAD & CLEAN DATA ----------------
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

# ---------------- OPENAI CONFIG ----------------
openai.api_key = "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"  # <-- Replace with your key

# ---------------- SIDEBAR AI TOOL ----------------
st.sidebar.title("🤖 AI Assistant")
user_question = st.sidebar.text_input("Ask me anything:")

def ai_answer(question):
    question = str(question).strip()
    if question == "":
        return ""

    q_lower = question.lower()

    # -------- Laptop price query (like "laptops under 60000") --------
    price_match = re.search(r'under\s*₹?(\d+)', q_lower)
    if price_match:
        max_price = int(price_match.group(1))
        filtered = df[df["Price"] <= max_price].sort_values(["Rating","Price"], ascending=[False,True])
        if filtered.empty:
            return f"No laptops found under ₹{max_price}"
        response = f"💻 Laptops under ₹{max_price}:\n"
        for i, row in filtered.head(5).iterrows():
            response += f"- {row['Model']} | Price: ₹{row['Price']} | RAM: {row['Ram']} | SSD: {row['SSD']} | Rating: {row['Rating']}\n"
        return response

    # -------- Specific model query --------
    models_lower = [m.lower() for m in df["Model"].tolist()]
    for model in models_lower:
        if model in q_lower:
            row = df[df["Model"].str.lower()==model].iloc[0]
            return f"💡 {row['Model']} recommended because Price ₹{row['Price']}, RAM {row['Ram']}, SSD {row['SSD']}, Rating {row['Rating']}, Graphics {row['Graphics']}"

    # -------- General AI response --------
    laptop_context = df.head(20).to_string()
    prompt = f"""
You are a helpful AI assistant. Answer any question the user asks.
Use the laptop dataset below only if the question is about laptops:

Laptop dataset:
{laptop_context}

User question: {question}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a friendly AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, I could not answer due to: {str(e)}"

if user_question:
    st.sidebar.markdown("**Answer:**")
    st.sidebar.info(ai_answer(user_question))

# ---------------- UI ----------------
st.title("💻 Laptop Finder AI")
st.caption("Smart Laptop Recommendation System with Explainable AI")

tabs = st.tabs([
    "🔍 Recommend",
    "🔎 Search",
    "💰 Price Filter",
    "🧠 Smart Laptop Insights",
    "📈 Trending Laptops"
])

# ================== TAB 1: RECOMMEND ==================
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
            score = max(0, 100 - dist[0][i-1]*10)
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
            st.markdown(f"""
            **{i}. {row['Model']}**  
            💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  
            🎮 {row['Graphics']} | ⭐ {row['Rating']}  
            🔍 Match Score: {score:.1f}%  
            🛒 [Buy on Amazon]({link})
            """)

# ================== TAB 2: SEARCH ==================
with tabs[1]:
    query = st.text_input("Search Laptop (Brand / Model)")
    if query:
        results = df[df["Model"].str.contains(query, case=False, na=False)]
        if results.empty:
            st.warning("No laptops found.")
        else:
            for _, row in results.head(10).iterrows():
                st.markdown(f"""
                **{row['Model']}**  
                💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']} | ⭐ {row['Rating']}
                """)

# ================== TAB 3: PRICE FILTER ==================
with tabs[2]:
    min_p, max_p = st.slider(
        "Select Price Range (₹)",
        int(df["Price"].min()),
        int(df["Price"].max()),
        (30000, 80000),
        step=5000
    )
    filtered = df[(df["Price"]>=min_p)&(df["Price"]<=max_p)]
    filtered = filtered.sort_values(["Rating","Price"], ascending=[False,True]).head(20)
    for _, row in filtered.iterrows():
        st.markdown(f"""
        **{row['Model']}**  
        💰 ₹{row['Price']} | ⭐ {row['Rating']} | 💾 {row['Ram']}
        """)

# ================== TAB 4: SMART LAPTOP INSIGHTS ==================
with tabs[3]:
    laptop_name = st.selectbox("Select a Laptop", df["Model"].unique())
    row = df[df["Model"]==laptop_name].iloc[0]

    st.subheader("🔍 Explainable AI – Why This Laptop?")
    budget_score = min(100,(1 - row["Price"]/df["Price"].max())*100)
    ram_score = min(100,(row["Ram_GB"]/32)*100)
    ssd_score = min(100,(row["SSD_GB"]/1024)*100)
    rating_score = (row["Rating"]/5)*100
    overall_score = (budget_score + ram_score + ssd_score + rating_score)/4
    st.write(f"💡 **Overall Confidence Score:** {overall_score:.1f}%")

    st.subheader("💸 Value for Money")
    value_score = (row["Ram_GB"]*2 + row["SSD_GB"]/256 + row["Rating"]*5)/row["Price"]
    st.write(f"💰 **Value Score:** {value_score:.2f}")

    st.subheader("🧓 Longevity / Future-Proof Score")
    longevity = 0
    longevity += 40 if row["Ram_GB"]>=16 else 20
    longevity += 30 if row["SSD_GB"]>=512 else 15
    longevity += 20 if row["Graphics_Flag"]==1 else 10
    longevity += 10 if row["Rating"]>=4 else 5
    st.write(f"⏳ **Future-Proof Score:** {longevity}/100")

    st.subheader("🎯 Usage Fit Score")
    st.write(f"🎮 Gaming Fit: {85 if row['Graphics_Flag']==1 else 60}%")
    st.write(f"💻 Programming Fit: {90 if row['Ram_GB']>=8 else 65}%")
    st.write(f"🎬 Editing Fit: {88 if row['SSD_GB']>=512 else 60}%")
    st.write(f"📄 Office Fit: 92%")

    st.subheader("💡 Upgrade Advice")
    if row["Ram_GB"]<16:
        st.warning("Upgrade RAM to 16GB for better performance.")
    if row["SSD_GB"]<512:
        st.warning("Upgrade SSD to 512GB for faster speed.")
    if row["Graphics_Flag"]==0:
        st.info("Not suitable for heavy gaming or video editing.")

# ================== TAB 5: TRENDING / POPULAR LAPTOPS ==================
with tabs[4]:
    df["Trending_Score"] = ((df["Rating"]/5)*50 +
                            ((df["Ram_GB"]*2 + df["SSD_GB"]/256 + df["Rating"]*5)/df["Price"]*30) +
                            ((df["Ram_GB"]>=16)*40 + (df["SSD_GB"]>=512)*30 + (df["Graphics_Flag"]==1)*20))
    trending = df.sort_values("Trending_Score", ascending=False).head(10)
    st.subheader("📈 Top 10 Trending Laptops")
    for _, row in trending.iterrows():
        link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
        st.markdown(f"""
        **{row['Model']}**  
        💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  
        🎮 {row['Graphics']} | ⭐ {row['Rating']}  
        🔥 Trending Score: {row['Trending_Score']:.1f}  
        🛒 [Buy on Amazon]({link})
        """)
