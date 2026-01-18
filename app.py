import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse
import google.generativeai as genai

# ================= Gemini API Key =================
genai.configure(api_key="AIzaSyAswyTKSinwbffaYQM3YE9VeBOdcSK0iRo")

st.set_page_config(page_title="Laptop Finder AI", layout="wide")

# ================= LOAD & CLEAN DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Clean price
    df["Price"] = df["Price"].str.replace("₹","").str.replace(",","").astype(float)
    df["Price"] = df["Price"].fillna(df["Price"].median())
    
    # RAM and SSD
    df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").fillna(0).astype(int)
    df["SSD_GB"] = df["SSD"].str.extract(r"(\d+)").fillna(0).astype(int)

    # GPU Flag
    df["Graphics_Flag"] = df["Graphics"].apply(lambda x: 0 if ("Intel" in str(x) or "UHD" in str(x) or "Iris" in str(x)) else 1)
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

    return df

df = load_data()

# ================= ML MODEL =================
X = df[["Price","Ram_GB","SSD_GB","Rating","Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# ================= AI ANSWER FUNCTION =================
def ai_answer(question):
    question = str(question).lower()
    
    # Dataset-based fallback
    response = "Sorry, I cannot answer that."

    if "gaming" in question:
        best = df[df["Graphics_Flag"]==1].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        response = f"🎮 Best gaming laptop: {best['Model']} with {best['Ram']} RAM, {best['SSD']} SSD, Rating {best['Rating']}"
    elif "student" in question:
        best = df[(df["Price"]<=60000) & (df["Rating"]>=3)].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        response = f"📚 Best student laptop: {best['Model']} with {best['Ram']} RAM, {best['SSD']} SSD, Price ₹{best['Price']}"
    elif any(l.lower() in question for l in df["Model"].tolist()):
        model = [l for l in df["Model"].tolist() if l.lower() in question][0]
        r = df[df["Model"]==model].iloc[0]
        response = f"💡 {r['Model']} recommended because Price ₹{r['Price']}, RAM {r['Ram']}, SSD {r['SSD']}, Rating {r['Rating']}, Graphics {r['Graphics']}"
    
    try:
        # Gemini API call
        gemini_resp = genai.chat.get_response(
            model="models/text-bison-001",
            prompt=question,
            temperature=0.7,
            candidate_count=1
        )
        ai_text = gemini_resp.last
        return ai_text
    except:
        # fallback to dataset answer
        return response

# ================= UI =================
st.title("💻 Laptop Finder AI")
st.caption("Smart Laptop Recommendation System with Gemini AI")

# ================= TABS =================
tabs = st.tabs([
    "🔍 Recommend",
    "🔎 Search",
    "💰 Price Filter",
    "🧠 Smart Laptop Insights",
    "📈 Trending Laptops"
])

# =====================================================
# TAB 1: RECOMMEND
# =====================================================
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        budget = st.slider("Budget (₹)", 20000, 200000, 60000, step=5000)
        ram = st.selectbox("RAM (GB)", [4,8,16,32])
        ssd = st.selectbox("SSD (GB)", [256,512,1024])
    with col2:
        rating = st.slider("Minimum Rating", 0.0,5.0,3.0,step=0.1)
        graphics = st.radio("Graphics", ["Integrated","Dedicated"])
    if st.button("Find Best Laptops"):
        g_flag = 0 if graphics=="Integrated" else 1
        user_input = scaler.transform([[budget,ram,ssd,rating,g_flag]])
        dist, idxs = knn.kneighbors(user_input)
        for i, idx in enumerate(idxs[0],1):
            row = df.iloc[idx]
            score = max(0,100 - dist[0][i-1]*10)
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
            st.markdown(f"**{i}. {row['Model']}**  💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  🎮 {row['Graphics']} | ⭐ {row['Rating']}  🔍 Match Score: {score:.1f}%  🛒 [Buy on Amazon]({link})")

# =====================================================
# TAB 2: SEARCH
# =====================================================
with tabs[1]:
    query = st.text_input("Search Laptop (Brand / Model)")
    if query:
        results = df[df["Model"].str.contains(query, case=False, na=False)]
        if results.empty:
            st.warning("No laptops found.")
        else:
            for _, row in results.head(10).iterrows():
                st.markdown(f"**{row['Model']}**  💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']} | ⭐ {row['Rating']}")

# =====================================================
# TAB 3: PRICE FILTER
# =====================================================
with tabs[2]:
    min_p,max_p = st.slider("Select Price Range (₹)", int(df["Price"].min()), int(df["Price"].max()), (30000,80000), step=5000)
    filtered = df[(df["Price"]>=min_p)&(df["Price"]<=max_p)]
    filtered = filtered.sort_values(["Rating","Price"], ascending=[False,True]).head(20)
    for _, row in filtered.iterrows():
        st.markdown(f"**{row['Model']}**  💰 ₹{row['Price']} | ⭐ {row['Rating']} | 💾 {row['Ram']}")

# =====================================================
# TAB 4: SMART LAPTOP INSIGHTS
# =====================================================
with tabs[3]:
    laptop_name = st.selectbox("Select a Laptop", df["Model"].unique())
    row = df[df["Model"]==laptop_name].iloc[0]

    st.subheader("🔍 Explainable AI – Why This Laptop?")
    budget_score = min(100,(1-row["Price"]/df["Price"].max())*100)
    ram_score = min(100,(row["Ram_GB"]/32)*100)
    ssd_score = min(100,(row["SSD_GB"]/1024)*100)
    rating_score = (row["Rating"]/5)*100
    overall_score = (budget_score+ram_score+ssd_score+rating_score)/4
    st.write(f"💡 **Overall Confidence Score:** {overall_score:.1f}%")

    st.subheader("💸 Value for Money")
    value_score = (row["Ram_GB"]*2 + row["SSD_GB"]/256 + row["Rating"]*5)/row["Price"]
    st.write(f"💰 **Value Score:** {value_score:.2f}")

# =====================================================
# TAB 5: TRENDING
# =====================================================
with tabs[4]:
    df["Trending_Score"] = ((df["Rating"]/5)*50 + ((df["Ram_GB"]*2 + df["SSD_GB"]/256 + df["Rating"]*5)/df["Price"]*30) + ((df["Ram_GB"]>=16)*40 + (df["SSD_GB"]>=512)*30 + (df["Graphics_Flag"]==1)*20))
    trending = df.sort_values("Trending_Score", ascending=False).head(10)
    for _, row in trending.iterrows():
        link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
        st.markdown(f"**{row['Model']}**  💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  🎮 {row['Graphics']} | ⭐ {row['Rating']}  🔥 Trending Score: {row['Trending_Score']:.1f}  🛒 [Buy on Amazon]({link})")

# ================= GEMINI FLOATING CHAT =================
from PIL import Image
gemini_logo = Image.open("gemini_logo.png")  # Add Gemini logo in folder
st.markdown("""
<style>
#chat-box {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    width: 90%; max-width: 700px; background-color: #1e1e1e; padding: 10px 15px;
    border-radius: 25px; display: flex; align-items: center; box-shadow: 0 4px 10px rgba(0,0,0,0.3); z-index:9999;
}
#chat-box input { flex:1; border:none; outline:none; background:transparent; color:white; font-size:16px; }
#chat-box button { background-color:#2b6cf2; border:none; color:white; padding:8px 12px; border-radius:15px; cursor:pointer; margin-left:8px; }
</style>
<div id="chat-box">
    <img src="gemini_logo.png" style="height:24px;margin-right:8px;">
    <input id="chat-input" placeholder="Ask Gemini...">
    <button onclick="sendMessage()">➤</button>
</div>
<script>
function sendMessage() {
    const input = document.getElementById('chat-input');
    if(input.value.trim()!=''){window.parent.postMessage({type:'streamlit:input', value:input.value},"*");input.value='';}
}
</script>
""", unsafe_allow_html=True)

# Hidden text input to capture JS message
user_question = st.text_input("", key="hidden_input")
if user_question:
    reply = ai_answer(user_question)
    st.chat_message("user").markdown(user_question)
    st.chat_message("assistant").markdown(reply)
