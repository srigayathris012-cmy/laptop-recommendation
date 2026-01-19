import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LaptopAI - Smart Laptop Recommendations",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Poppins:wght@600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content container with glass effect */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        color: #1f2937;
    }
    
    .block-container h1, .block-container h2, .block-container h3, .block-container h4 {
        color: #111827;
    }
    
    .block-container p, .block-container span, .block-container div {
        color: #374151;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .laptop-card {
        background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #1f2937;
    }
    
    .laptop-card h3, .laptop-card h4 {
        color: #111827;
    }
    
    .laptop-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #667eea;
        background-color: white;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Score badge */
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 10px rgba(16, 185, 129, 0.3);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Feature icons */
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

# ================= SIDEBAR AI ASSISTANT =================
st.sidebar.markdown("<h2 style='text-align: center;'>🤖 AI Assistant</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
user_question = st.sidebar.text_input("💬 Ask me anything about laptops:", placeholder="e.g., best gaming laptop under 60k")

def get_laptop_context():
    """Prepare laptop data context for AI"""
    # Get sample of laptops for context
    top_laptops = df.nlargest(10, 'Rating')[['Model', 'Price', 'Ram', 'SSD', 'Graphics', 'Rating']]
    budget_laptops = df[df['Price'] <= 60000].nlargest(5, 'Rating')[['Model', 'Price', 'Ram', 'SSD', 'Rating']]
    gaming_laptops = df[df['Graphics_Flag'] == 1].nlargest(5, 'Rating')[['Model', 'Price', 'Ram', 'SSD', 'Graphics', 'Rating']]
    
    context = f"""
You are a laptop recommendation expert. Here's information about our laptop database:

Total laptops available: {len(df)}
Price range: ₹{df['Price'].min():,} to ₹{df['Price'].max():,}
Average price: ₹{df['Price'].mean():,.0f}

TOP RATED LAPTOPS:
{top_laptops.to_string(index=False)}

BEST BUDGET LAPTOPS (Under ₹60,000):
{budget_laptops.to_string(index=False)}

BEST GAMING LAPTOPS:
{gaming_laptops.to_string(index=False)}

Please provide helpful, specific recommendations based on the user's question. Include model names, prices, and key specs when relevant.
"""
    return context

async def ai_answer_smart(question):
    """AI-powered answer using Claude API"""
    try:
        import aiohttp
        
        context = get_laptop_context()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{context}\n\nUser Question: {question}\n\nProvide a helpful, concise answer with specific laptop recommendations when appropriate."
                        }
                    ]
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = ""
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            answer += block.get("text", "")
                    return answer if answer else "I apologize, but I couldn't generate a response. Please try again."
                else:
                    # Fallback to basic answers
                    return get_basic_answer(question)
    except Exception as e:
        # Fallback to basic answers
        return get_basic_answer(question)

def get_basic_answer(question):
    """Fallback basic answers"""
    q = str(question).lower()
    
    if "gaming" in q:
        best = df[df["Graphics_Flag"] == 1].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"🎮 **Best gaming laptop:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nThis laptop has dedicated graphics which is perfect for gaming!"
    
    elif "student" in q or "budget" in q:
        best = df[(df["Price"] <= 60000) & (df["Rating"] >= 3)].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"📚 **Best student laptop:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nGreat for students with good specs at an affordable price!"
    
    elif "value" in q or "money" in q:
        df_temp = df.copy()
        df_temp["Value_Score"] = (df_temp["Ram_GB"]*2 + df_temp["SSD_GB"]/256 + df_temp["Rating"]*5)/df_temp["Price"]
        best = df_temp.sort_values("Value_Score", ascending=False).iloc[0]
        return f"💰 **Best value for money:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n📊 Value Score: {best['Value_Score']:.4f}\n\nThis gives you the most features for your money!"
    
    elif "program" in q or "coding" in q or "developer" in q:
        best = df[(df["Ram_GB"] >= 8) & (df["SSD_GB"] >= 512)].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"💻 **Best for programming:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nGreat specs for coding with sufficient RAM and fast SSD!"
    
    elif "video" in q or "editing" in q or "creative" in q:
        best = df[(df["Graphics_Flag"] == 1) & (df["SSD_GB"] >= 512)].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"🎬 **Best for video editing:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nDedicated graphics and large storage perfect for editing!"
    
    elif "future" in q or "longevity" in q:
        best = df[(df["Ram_GB"] >= 16) & (df["SSD_GB"] >= 512) & (df["Graphics_Flag"] == 1)].sort_values(["Rating","Price"], ascending=[False,True])
        if len(best) > 0:
            best = best.iloc[0]
            return f"🔮 **Most future-proof:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nHigh RAM, large SSD, and dedicated graphics for years to come!"
    
    elif "cheap" in q or "affordable" in q or "low price" in q:
        best = df[df["Price"] <= 40000].sort_values(["Rating","Price"], ascending=[False,True])
        if len(best) > 0:
            best = best.iloc[0]
            return f"💵 **Most affordable option:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nBest rated laptop in the budget category!"
    
    elif "office" in q or "work" in q or "business" in q:
        best = df[(df["Ram_GB"] >= 8) & (df["Price"] <= 70000)].sort_values(["Rating","Price"], ascending=[False,True]).iloc[0]
        return f"💼 **Best for office work:** {best['Model']}\n\n💰 ₹{best['Price']:,} | 💾 {best['Ram']} | 💿 {best['SSD']}\n⭐ Rating: {best['Rating']}\n\nPerfect for productivity with good specs at reasonable price!"
    
    elif "price" in q and any(word in q for word in ["how much", "cost", "expensive"]):
        avg_price = df["Price"].mean()
        min_price = df["Price"].min()
        max_price = df["Price"].max()
        return f"💰 **Price Information:**\n\n📊 Average: ₹{avg_price:,.0f}\n📉 Lowest: ₹{min_price:,}\n📈 Highest: ₹{max_price:,}\n\nMost laptops range from ₹30,000 to ₹1,00,000"
    
    elif "?" in q or "recommend" in q:
        top_rated = df.nlargest(3, 'Rating')
        response = "🏆 **Top 3 Recommended Laptops:**\n\n"
        for idx, (_, laptop) in enumerate(top_rated.iterrows(), 1):
            response += f"{idx}. **{laptop['Model']}**\n   💰 ₹{laptop['Price']:,} | ⭐ {laptop['Rating']}\n\n"
        return response
    
    return f"🤔 I can help you find the perfect laptop! Try asking:\n\n• 'Best gaming laptop'\n• 'Laptop for students under 60k'\n• 'Best value for money'\n• 'Laptop for programming'\n• 'Future-proof laptop'\n\nOr use the tabs above to explore options!"

if user_question:
    st.sidebar.markdown("### 💡 Answer")
    with st.sidebar:
        with st.spinner("🤔 Thinking..."):
            # Try to use AI, fallback to basic if it fails
            try:
                import asyncio
                answer = asyncio.run(ai_answer_smart(user_question))
                st.success(answer)
            except:
                # If async fails, use basic answers
                answer = get_basic_answer(user_question)
                st.success(answer)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.metric("Total Laptops", len(df))
st.sidebar.metric("Avg Price", f"₹{df['Price'].mean():,.0f}")
st.sidebar.metric("Top Rating", f"⭐ {df['Rating'].max()}")

# ================= MAIN HEADER =================
st.markdown("<h1 class='main-title'>💻 LaptopAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>🚀 Intelligent Laptop Recommendations Powered by Machine Learning</p>", unsafe_allow_html=True)

# ================= TABS =================
tabs = st.tabs([
    "🔍 Smart Recommend",
    "🔎 Search",
    "💰 Price Filter",
    "🧠 Deep Insights",
    "📈 Trending"
])

# =====================================================
# TAB 1: SMART RECOMMEND
# =====================================================
with tabs[0]:
    st.markdown("### 🎯 Find Your Perfect Laptop")
    st.markdown("Answer a few questions and let our AI find the best match for you!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        budget = st.slider("💵 Budget (₹)", 20000, 200000, 60000, step=5000, help="Set your maximum budget")
        ram = st.selectbox("💾 RAM (GB)", [4, 8, 16, 32], index=1)
    
    with col2:
        ssd = st.selectbox("💿 Storage (GB)", [256, 512, 1024], index=1)
        rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0, step=0.1)
    
    with col3:
        graphics = st.radio("🎮 Graphics", ["Integrated", "Dedicated"], help="Dedicated for gaming/editing")
        st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Find Best Laptops", use_container_width=True):
        g_flag = 0 if graphics=="Integrated" else 1
        user_input = scaler.transform([[budget, ram, ssd, rating, g_flag]])
        dist, idxs = knn.kneighbors(user_input)
        
        st.markdown("### 🏆 Top Recommendations")
        
        for i, idx in enumerate(idxs[0], 1):
            row = df.iloc[idx]
            score = max(0, 100 - dist[0][i-1]*10)
            link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
            
            # Color-coded ranking
            rank_color = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            
            st.markdown(f"""
            <div class='laptop-card'>
                <h3 style='color: #111827;'>{rank_color} {row['Model']}</h3>
                <div style='display: flex; justify-content: space-between; flex-wrap: wrap; margin: 1rem 0; color: #374151;'>
                    <span><strong>💰 Price:</strong> ₹{row['Price']:,}</span>
                    <span><strong>💾 RAM:</strong> {row['Ram']}</span>
                    <span><strong>💿 Storage:</strong> {row['SSD']}</span>
                    <span><strong>⭐ Rating:</strong> {row['Rating']}</span>
                </div>
                <div style='margin: 1rem 0; color: #374151;'>
                    <strong>🎮 Graphics:</strong> {row['Graphics']}
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span class='score-badge'>🎯 Match: {score:.1f}%</span>
                    <a href='{link}' target='_blank' style='text-decoration: none;'>
                        <button style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: none; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer;'>
                            🛒 View on Amazon
                        </button>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =====================================================
# TAB 2: SEARCH
# =====================================================
with tabs[1]:
    st.markdown("### 🔎 Search for Specific Laptops")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("", placeholder="Search by brand or model name (e.g., Dell, HP, Lenovo)")
    
    if query:
        results = df[df["Model"].str.contains(query, case=False, na=False)]
        
        if results.empty:
            st.warning("🔍 No laptops found. Try a different search term.")
        else:
            st.success(f"✅ Found {len(results)} laptop(s)")
            
            for _, row in results.head(10).iterrows():
                link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
                st.markdown(f"""
                <div class='laptop-card'>
                    <h4 style='color: #111827;'>{row['Model']}</h4>
                    <div style='display: flex; justify-content: space-between; margin: 1rem 0; color: #374151;'>
                        <span>💰 ₹{row['Price']:,}</span>
                        <span>💾 {row['Ram']}</span>
                        <span>💿 {row['SSD']}</span>
                        <span>⭐ {row['Rating']}</span>
                    </div>
                    <a href='{link}' target='_blank' style='text-decoration: none; color: #667eea; font-weight: 600;'>
                        🛒 View Details →
                    </a>
                </div>
                """, unsafe_allow_html=True)

# =====================================================
# TAB 3: PRICE FILTER
# =====================================================
with tabs[2]:
    st.markdown("### 💰 Filter by Price Range")
    
    min_p, max_p = st.slider(
        "Select your price range:",
        int(df["Price"].min()),
        int(df["Price"].max()),
        (30000, 80000),
        step=5000
    )
    
    filtered = df[(df["Price"]>=min_p)&(df["Price"]<=max_p)]
    filtered = filtered.sort_values(["Rating","Price"], ascending=[False,True]).head(20)
    
    st.info(f"📊 Showing {len(filtered)} laptops between ₹{min_p:,} and ₹{max_p:,}")
    
    for _, row in filtered.iterrows():
        link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
        st.markdown(f"""
        <div class='laptop-card'>
            <h4 style='color: #111827;'>{row['Model']}</h4>
            <div style='display: flex; justify-content: space-between; align-items: center; color: #374151;'>
                <div>
                    <span style='font-size: 1.3rem; font-weight: 700; color: #667eea;'>₹{row['Price']:,}</span>
                    <span style='margin-left: 1rem;'>⭐ {row['Rating']}</span>
                    <span style='margin-left: 1rem;'>💾 {row['Ram']}</span>
                </div>
                <a href='{link}' target='_blank' style='text-decoration: none; color: #667eea; font-weight: 600;'>
                    View →
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# TAB 4: DEEP INSIGHTS
# =====================================================
with tabs[3]:
    st.markdown("### 🧠 AI-Powered Laptop Analysis")
    
    laptop_name = st.selectbox("Choose a laptop to analyze:", df["Model"].unique())
    row = df[df["Model"]==laptop_name].iloc[0]
    
    # Header with laptop name
    st.markdown(f"<h3 class='gradient-text'>{laptop_name}</h3>", unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Price</div>
            <div class='metric-value'>₹{row['Price']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Rating</div>
            <div class='metric-value'>⭐ {row['Rating']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>RAM</div>
            <div class='metric-value'>{row['Ram_GB']}GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Storage</div>
            <div class='metric-value'>{row['SSD_GB']}GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overall Score
    budget_score = min(100,(1 - row["Price"]/df["Price"].max())*100)
    ram_score = min(100,(row["Ram_GB"]/32)*100)
    ssd_score = min(100,(row["SSD_GB"]/1024)*100)
    rating_score = (row["Rating"]/5)*100
    overall_score = (budget_score + ram_score + ssd_score + rating_score)/4
    
    st.markdown("#### 🎯 Overall Confidence Score")
    st.progress(min(1.0, max(0.0, overall_score/100)))
    st.markdown(f"<h2 style='text-align: center; color: #667eea;'>{overall_score:.1f}%</h2>", unsafe_allow_html=True)
    
    # Two column layout for insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Value for Money")
        value_score = (row["Ram_GB"]*2 + row["SSD_GB"]/256 + row["Rating"]*5)/row["Price"]
        st.metric("Value Score", f"{value_score:.4f}", help="Higher is better")
        
        st.markdown("#### 🔮 Future-Proof Score")
        longevity = 0
        longevity += 40 if row["Ram_GB"]>=16 else 20
        longevity += 30 if row["SSD_GB"]>=512 else 15
        longevity += 20 if row["Graphics_Flag"]==1 else 10
        longevity += 10 if row["Rating"]>=4 else 5
        st.progress(min(1.0, max(0.0, longevity/100)))
        st.markdown(f"**{longevity}/100** - " + ("Excellent" if longevity>=80 else "Good" if longevity>=60 else "Moderate"))
    
    with col2:
        st.markdown("#### 🎯 Usage Fit Scores")
        gaming_fit = 85 if row['Graphics_Flag']==1 else 60
        prog_fit = 90 if row['Ram_GB']>=8 else 65
        edit_fit = 88 if row['SSD_GB']>=512 else 60
        
        st.markdown(f"🎮 **Gaming:** {gaming_fit}%")
        st.progress(min(1.0, max(0.0, gaming_fit/100)))
        
        st.markdown(f"💻 **Programming:** {prog_fit}%")
        st.progress(min(1.0, max(0.0, prog_fit/100)))
        
        st.markdown(f"🎬 **Video Editing:** {edit_fit}%")
        st.progress(min(1.0, max(0.0, edit_fit/100)))
        
        st.markdown(f"📄 **Office Work:** 92%")
        st.progress(0.92)
    
    # Upgrade recommendations
    st.markdown("#### 💡 Smart Recommendations")
    if row["Ram_GB"]<16:
        st.warning("⚠️ Consider upgrading RAM to 16GB for better multitasking performance.")
    if row["SSD_GB"]<512:
        st.warning("⚠️ Upgrade to 512GB SSD for faster boot times and more storage.")
    if row["Graphics_Flag"]==0:
        st.info("ℹ️ This laptop has integrated graphics - not ideal for heavy gaming or video editing.")
    if row["Ram_GB"]>=16 and row["SSD_GB"]>=512:
        st.success("✅ This laptop has excellent specifications!")

# =====================================================
# TAB 5: TRENDING LAPTOPS
# =====================================================
with tabs[4]:
    st.markdown("### 📈 Top Trending Laptops")
    st.markdown("Based on ratings, value, and specifications")
    
    df["Trending_Score"] = ((df["Rating"]/5)*50 +
                            ((df["Ram_GB"]*2 + df["SSD_GB"]/256 + df["Rating"]*5)/df["Price"]*30) +
                            ((df["Ram_GB"]>=16)*40 + (df["SSD_GB"]>=512)*30 + (df["Graphics_Flag"]==1)*20))
    trending = df.sort_values("Trending_Score", ascending=False).head(10)
    
    for idx, (_, row) in enumerate(trending.iterrows(), 1):
        link = "https://www.amazon.in/s?k=" + urllib.parse.quote(row["Model"])
        
        medal = ["🥇", "🥈", "🥉"][idx-1] if idx <= 3 else f"**#{idx}**"
        
        st.markdown(f"""
        <div class='laptop-card'>
            <h3 style='color: #111827;'>{medal} {row['Model']}</h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin: 1rem 0; color: #374151;'>
                <div><strong>💰 Price:</strong> ₹{row['Price']:,}</div>
                <div><strong>💾 RAM:</strong> {row['Ram']}</div>
                <div><strong>💿 Storage:</strong> {row['SSD']}</div>
                <div><strong>⭐ Rating:</strong> {row['Rating']}</div>
            </div>
            <div style='margin: 1rem 0; color: #374151;'>
                <strong>🎮 Graphics:</strong> {row['Graphics']}
            </div>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span class='score-badge'>🔥 Trending Score: {row['Trending_Score']:.1f}</span>
                <a href='{link}' target='_blank' style='text-decoration: none;'>
                    <button style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 600; cursor: pointer;'>
                        🛒 Buy Now
                    </button>
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p style='font-size: 0.9rem;'>Made with ❤️ using Streamlit & Machine Learning</p>
    <p style='font-size: 0.8rem;'>© 2024 LaptopAI - Your Intelligent Laptop Shopping Assistant</p>
</div>
""", unsafe_allow_html=True)
