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
rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 3.5, 0.1)
graphics = st.sidebar.radio("🎮 Dedicated Graphics?", ["No", "Yes"])
graphics_input = 1 if graphics == "Yes" else 0

# -----------------------------
# Recommendations
# -----------------------------
# -----------------------------
# Recommendations
# -----------------------------
if st.sidebar.button("🔍 Recommend Laptops"):

    user_input_dict = {
        "Price": budget_value,
        "Ram_GB": ram,
        "SSD_GB": ssd,
        "Rating": rating,
        "Graphics_Flag": graphics_input
    }

    user_input = [[user_input_dict[col] for col in feature_cols]]
    user_scaled = scaler.transform(user_input)

    distances, indices = knn.kneighbors(user_scaled)

    rec_df = df_display.iloc[indices[0]]
    rec_df = rec_df.drop_duplicates(subset="Model").head(5)

    st.subheader("✅ Recommended Laptops")

    for _, row in rec_df.iterrows():
        st.markdown(
            f"""
            <div style="background-color:#ffffff;
                        color:#000000;
                        padding:15px;
                        border-radius:10px;
                        margin-bottom:15px;
                        border:1px solid #ddd;">
                <h3>{row['Model']}</h3>
                <p><b>Price:</b> ₹{row['Price']}</p>
                <p><b>RAM:</b> {row['Ram']}</p>
                <p><b>SSD:</b> {row['SSD']}</p>
                <p><b>Graphics:</b> {row.get('Graphics','N/A')}</p>
                <p><b>Rating:</b> ⭐ {row.get('Rating','N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
