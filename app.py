import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse
from PIL import Image

st.set_page_config(page_title="Laptop Finder AI", layout="wide")

# ================= LOAD & CLEAN DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    def clean_price(x):
        try:
            return int(str(x).replace("₹","").replace(",","").strip())
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
X = df[["Price","Ram_GB","SSD_GB","Rating","Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X_scaled)

# ================= AI FUNCTION =================
def ai_answer(question):
    q = str(question).lower()
    response = "Sorry, I cannot answer that."

    if "gaming" in q:
        best = df[df["Graphics_Flag"]==1].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        response = f"🎮 Best gaming laptop: {best['Model']} with {best['Ram']} RAM, {best['SSD']} SSD, Rating {best['Rating']}"
    elif "student" in q:
        best = df[(df["Price"]<=60000) & (df["Rating"]>=3)].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        response = f"📚 Best student laptop: {best['Model']} with {best['Ram']} RAM, {best['SSD']} SSD, Price ₹{best['Price']}"
    elif "value" in q:
        df["Value_Score"] = (df["Ram_GB"]*2 + df["SSD_GB"]/256 + df["Rating"]*5)/df["Price"]
        best = df.sort_values("Value_Score", ascending=False).iloc[0]
        response = f"💰 Best value for money: {best['Model']} Value Score: {best['Value_Score']:.2f}"
    elif "longevity" in q or "future" in q:
        scores = []
        for _, r in df.iterrows():
            score = 0
            score += 40 if r["Ram_GB"]>=16 else 20
            score += 30 if r["SSD_GB"]>=512 else 15
            score += 20 if r["Graphics_Flag"]==1 else 10
            scores.append(score)
        df["Longevity_Score"] = scores
        best = df.sort_values("Longevity_Score", ascending=False).iloc[0]
        response = f"🧓 Most future-proof laptop: {best['Model']} Longevity Score: {best['Longevity_Score']}"
    elif any(l.lower() in q for l in df["Model"].tolist()):
        model = [l for l in df["Model"].tolist() if l.lower() in q][0]
        r = df[df["Model"]==model].iloc[0]
        response = f"💡 {r['Model']} recommended because Price ₹{r['Price']}, RAM {r['Ram']}, SSD {r['SSD']}, Rating {r['Rating']}, Graphics {r['Graphics']}"
    return response

# ================= GEMINI SIDEBAR =================
st.sidebar.title("🤖 Ask Gemini")
try:
    gemini_logo = Image.open("gemini_logo.png")  # Place PNG logo in same folder
    st.sidebar.image(gemini_logo, width=50)
except:
    st.sidebar.warning("Gemini logo not found. Place gemini_logo.png in the app folder.")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.sidebar.text_input("Ask Gemini:")
if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    st.session_state.messages.append({"role":"assistant","content":ai_answer(user_input)})

for msg in st.session_state.messages:
    if msg["role"]=="user":
        st.sidebar.markdown(f"**You:** {msg['content']}")
    else:
        st.sidebar.markdown(f"**Gemini:** {msg['content']}")

# ================= UI =================
st.title("💻 Laptop Finder AI")
st.caption("Smart Laptop Recommendation System with Explainable AI")

tabs = st.tabs([
    "🔍 Recommend",
    "🔎 Search",
    "💰 Price Filter",
    "🧠 Smart Laptop Insights",
    "📈 Trending Laptops"
])

# ---------------- TAB 1: RECOMMEND ----------------
with tabs[0]:
    col1,col2 = st.columns(2)
    with col1:
        budget = st.slider("Budget (₹)",20000,200000,60000,step=5000)
        ram = st.selectbox("RAM (GB)", [4,8,16,32])
        ssd = st.selectbox("SSD (GB)", [256,512,1024])
    with col2:
        rating = st.slider("Minimum Rating",0.0,5.0,3.0,step=0.1)
        graphics = st.radio("Graphics", ["Integrated","Dedicated"])
    if st.button("Find Best Laptops"):
        g_flag = 0 if graphics=="Integrated" else 1
        user_input_scaled = scaler.transform([[budget, ram, ssd, rating, g_flag]])
        dist, idxs = knn.kneighbors(user_input_scaled)
        for i, idx in enumerate(idxs[0],1):
            row = df.iloc[idx]
            score = max(0, 100 - dist[0][i-1]*10)
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
            st.markdown(f"**{i}. {row['Model']}**  \n💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  \n🎮 {row['Graphics']} | ⭐ {row['Rating']}  \n🔍 Match Score: {score:.1f}%  \n🛒 [Buy on Amazon]({link})")

# ---------------- TAB 2: SEARCH ----------------
with tabs[1]:
    query = st.text_input("Search Laptop (Brand / Model)")
    if query:
        results = df[df["Model"].str.contains(query, case=False, na=False)]
        if results.empty:
            st.warning("No laptops found.")
        else:
            for _, row in results.head(10).iterrows():
                st.markdown(f"**{row['Model']}**  \n💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']} | ⭐ {row['Rating']}")

# ---------------- TAB 3: PRICE FILTER ----------------
with tabs[2]:
    min_p,max_p = st.slider("Select Price Range (₹)", int(df["Price"].min()), int(df["Price"].max()), (30000,80000), step=5000)
    filtered = df[(df["Price"]>=min_p)&(df["Price"]<=max_p)]
    filtered = filtered.sort_values(["Rating","Price"], ascending=[False,True]).head(20)
    for _, row in filtered.iterrows():
        st.markdown(f"**{row['Model']}**  \n💰 ₹{row['Price']} | ⭐ {row['Rating']} | 💾 {row['Ram']}")

# ---------------- TAB 4: SMART LAPTOP INSIGHTS ----------------
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

# ---------------- TAB 5: TRENDING ----------------
with tabs[4]:
    df["Trending_Score"] = ((df["Rating"]/5)*50 +
                            ((df["Ram_GB"]*2 + df["SSD_GB"]/256 + df["Rating"]*5)/df["Price"]*30) +
                            ((df["Ram_GB"]>=16)*40 + (df["SSD_GB"]>=512)*30 + (df["Graphics_Flag"]==1)*20))
    trending = df.sort_values("Trending_Score", ascending=False).head(10)
    st.subheader("📈 Top 10 Trending Laptops")
    for _, row in trending.iterrows():
        link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
        st.markdown(f"**{row['Model']}**  \n💰 ₹{row['Price']} | 💾 {row['Ram']} | 💿 {row['SSD']}  \n🎮 {row['Graphics']} | ⭐ {row['Rating']}  \n🔥 Trending Score: {row['Trending_Score']:.1f}  \n🛒 [Buy on Amazon]({link})")
