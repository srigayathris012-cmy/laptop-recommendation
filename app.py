import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("💻 Laptop Recommendation System")
st.caption("Get the best laptop based on your requirements")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    df.columns = df.columns.str.strip()
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

df = load_data()
df_display = df.copy()  # Keep original for display

# -----------------------------
# Clean Columns (Price, RAM, SSD, Rating)
# -----------------------------
df["Price"] = df["Price"].str.replace("₹","").str.replace(",","").astype(int)

df["Ram_GB"] = df["Ram"].str.extract("(\d+)").astype(float).fillna(0).astype(int)

def parse_ssd(x):
    try:
        if pd.isna(x):
            return 0
        x = str(x).upper()
        if "TB" in x:
            return int(float(x.replace("TB","").strip())*1024)
        else:
            return int(float(x.replace("GB","").strip()))
    except:
        return 0
df["SSD_GB"] = df["SSD"].apply(parse_ssd)

df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

# Graphics Flag
def graphics_flag(x):
    x = str(x)
    if "Intel" in x or "UHD" in x or "Iris" in x:
        return 0
    return 1
df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)

# -----------------------------
# ML Features
# -----------------------------
feature_cols = ["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]
X = df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn.fit(X_scaled)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🛠 Your Requirements")

budget_options = [
    "Below ₹30,000",
    "₹30,000 – ₹50,000",
    "₹50,000 – ₹70,000",
    "₹70,000 – ₹1,00,000",
    "Above ₹1,00,000"
]

budget = st.sidebar.selectbox("💰 Budget (₹)", budget_options, index=2)
budget_value = {
    "Below ₹30,000": 25000,
    "₹30,000 – ₹50,000": 40000,
    "₹50,000 – ₹70,000": 60000,
    "₹70,000 – ₹1,00,000": 85000,
    "Above ₹1,00,000": 120000
}[budget]

ram = st.sidebar.selectbox("🧠 RAM (GB)", [4, 8, 16, 32])
ssd = st.sidebar.selectbox("💾 SSD (GB)", [256, 512, 1024])
rating = st.sidebar.selectbox("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
graphics = st.sidebar.radio("🎮 Dedicated Graphics?", ["No", "Yes"])
graphics_input = 1 if graphics == "Yes" else 0


budget_value = budget_map[budget]

ram = st.sidebar.selectbox("🧠 RAM (GB)", [4, 8, 16, 32])
ssd = st.sidebar.selectbox("💾 SSD (GB)", [256, 512, 1024])
rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
graphics = st.sidebar.radio("🎮 Dedicated Graphics?", ["No", "Yes"])
graphics_input = 1 if graphics == "Yes" else 0

# -----------------------------
# Recommendations
# -----------------------------
if st.sidebar.button("🔍 Recommend Laptops"):
    user_input = [[budget, ram, ssd, rating, graphics_input]]
    user_scaled = scaler.transform(user_input)
    distances, indices = knn.kneighbors(user_scaled)

    rec_df = df_display.iloc[indices[0]].drop_duplicates(subset="Model").head(5)

    st.subheader("✅ Recommended Laptops")
for _, row in rec_df.iterrows():
    with st.container():
        st.markdown(
            f"""
            <div style="
                border:1px solid #555;
                border-radius:10px;
                padding:20px;
                margin-bottom:20px;
                background-color:#1e1e1e;
                color:white;
                box-shadow:2px 2px 15px rgba(0,0,0,0.5);
            ">
                <h3 style="color:#ffffff;">{row['Model']}</h3>
                <p><b>Price:</b> ₹{row.get('Price','N/A')}</p>
                <p><b>RAM:</b> {row.get('Ram','N/A')}</p>
                <p><b>SSD:</b> {row.get('SSD','N/A')}</p>
                <p><b>Graphics:</b> {row.get('Graphics','N/A')}</p>
                <p><b>Display:</b> {row.get('Display','N/A')}</p>
                <p><b>Rating:</b> ⭐ {row.get('Rating','N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
