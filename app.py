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
    df.columns = df.columns.str.strip()  # Remove extra spaces
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

df = load_data()
df_display = df.copy()  # Original for display

# -----------------------------
# Clean & Parse Columns
# -----------------------------
# Price
df["Price"] = df["Price"].str.replace("₹","").str.replace(",","").astype(int)

# RAM
def parse_ram(x):
    try:
        if pd.isna(x):
            return 0
        return int(str(x).split()[0])
    except:
        return 0
df["Ram_GB"] = df["Ram"].apply(parse_ram)

# SSD
def parse_ssd(x):
    try:
        if pd.isna(x):
            return 0
        x = str(x).upper()
        x_list = x.replace("GB","").replace("TB","").split("/")
        numbers = []
        for val in x_list:
            val = val.strip()
            if val == "":
                continue
            num = float(val)
            if "TB" in x:
                num *= 1024
            numbers.append(num)
        if len(numbers) == 0:
            return 0
        return int(max(numbers))
    except:
        return 0
df["SSD_GB"] = df["SSD"].apply(parse_ssd)

# Rating
if "Rating" in df.columns:
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
else:
    df["Rating"] = 0

# Graphics flag
if "Graphics" in df.columns:
    def graphics_flag(x):
        x = str(x)
        if "Intel" in x or "UHD" in x:
            return 0
        return 1
    df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)
else:
    df["Graphics_Flag"] = 0

# -----------------------------
# ML Features
# -----------------------------
feature_cols = ["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]
feature_cols = [col for col in feature_cols if col in df.columns]
X = df[feature_cols]

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
# Recommendations
# -----------------------------
if st.sidebar.button("🔍 Recommend Laptops"):
    user_input_dict = {
        "Price": budget,
        "Ram_GB": ram,
        "SSD_GB": ssd,
        "Rating": rating,
        "Graphics_Flag": graphics_input
    }
    user_input = [[user_input_dict[col] for col in feature_cols]]
    user_scaled = scaler.transform(user_input)
    distances, indices = knn.kneighbors(user_scaled)

    rec_df = df_display.iloc[indices[0]].drop_duplicates(subset="Model").head(5)

    st.subheader("✅ Recommended Laptops")
 for _, row in rec_df.iterrows():
    with st.container():
        st.markdown(
            f"""
            <div style="border:1px solid #444; padding:15px; border-radius:10px; margin-bottom:15px; background-color:#f9f9f9; color:black;">
                <h3 style="color:black;">{row['Model']}</h3>
                <p><b>Price:</b> {row.get('Price', 'N/A')}</p>
                <p><b>RAM:</b> {row.get('Ram', 'N/A')}</p>
                <p><b>SSD:</b> {row.get('SSD', 'N/A')}</p>
                <p><b>Graphics:</b> {row.get('Graphics', 'N/A')}</p>
                <p><b>Display:</b> {row.get('Display', 'N/A')}</p>
                <p><b>Rating:</b> ⭐ {row.get('Rating', 'N/A')}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
