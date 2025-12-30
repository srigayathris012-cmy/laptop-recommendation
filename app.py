import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("💻 Laptop Recommendation System")
st.caption("Get the best laptop based on your requirements")

@st.cache_data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

df = load_data()
df_display = df.copy()  # keep original for user output

# ----------------- CLEAN DATA -----------------
df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(int)

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
    """
    Converts SSD column to GB integer
    Handles:
      - '256 GB', '512 GB'
      - '1 TB' => 1024 GB
      - 'HDD', 'No SSD', NaN => 0
      - Multiple values like '256GB/512GB' => take the max
    """
    try:
        if pd.isna(x):
            return 0
        x = str(x).upper()
        # Handle multiple values
        x_list = x.replace("GB","").replace("TB","").split("/")
        numbers = []
        for val in x_list:
            val = val.strip()
            if val == "":
                continue
            num = float(val)
            # Convert TB to GB
            if "TB" in x:
                num *= 1024
            numbers.append(num)
        if len(numbers) == 0:
            return 0
        return int(max(numbers))
    except:
        return 0

df["SSD_GB"] = df["SSD"].apply(parse_ssd)

# ----------------- ML FEATURES -----------------
X = df[["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
knn.fit(X_scaled)

# ----------------- Sidebar -----------------
st.sidebar.header("🛠 Your Requirements")
budget = st.sidebar.slider("💰 Budget (₹)", 20000, 150000, 60000)
ram = st.sidebar.selectbox("🧠 RAM (GB)", [4, 8, 16, 32])
ssd = st.sidebar.selectbox("💾 SSD (GB)", [256, 512, 1024])
rating = st.sidebar.slider("⭐ Minimum Rating", 40, 100, 60)
graphics = st.sidebar.radio("🎮 Dedicated Graphics?", ["No", "Yes"])
graphics_input = 1 if graphics == "Yes" else 0

# ----------------- Recommendation -----------------
if st.sidebar.button("🔍 Recommend Laptops"):
    user_input = [[budget, ram, ssd, rating, graphics_input]]
    user_scaled = scaler.transform(user_input)
    distances, indices = knn.kneighbors(user_scaled)

    rec_df = df_display.iloc[indices[0]]
    rec_df = rec_df.drop_duplicates(subset="Model").head(5)

    st.subheader("✅ Recommended Laptops")
    for _, row in rec_df.iterrows():
        with st.container():
            st.markdown(
                f"""
                <div style="border:1px solid #444; padding:15px; border-radius:10px; margin-bottom:15px; background-color:#f9f9f9;">
                    <h3>{row['Model']}</h3>
                    <p><b>Price:</b> {row['Price']}</p>
                    <p><b>RAM:</b> {row['Ram']}</p>
                    <p><b>SSD:</b> {row['SSD']}</p>
                    <p><b>Graphics:</b> {row['Graphics']}</p>
                    <p><b>Display:</b> {row['Display']}</p>
                    <p><b>Rating:</b> ⭐ {row['Rating']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
