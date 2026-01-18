import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import urllib.parse

# ---------------- LOAD DATA ----------------
df = pd.read_csv("laptop.csv")
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df_display = df.copy()

# ---------------- CLEAN PRICE ----------------
def clean_price(x):
    try:
        return int(str(x).replace("₹","").replace(",","").strip())
    except:
        return np.nan

df["Price"] = df["Price"].apply(clean_price)
df = df.dropna(subset=["Price"])
df["Price"] = df["Price"].astype(int)

# ---------------- EXTRACT RAM & SSD ----------------
df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").fillna(0).astype(int)
df["SSD_GB"] = df["SSD"].str.extract(r"(\d+)").fillna(0).astype(int)

# ---------------- GRAPHICS FLAG ----------------
def graphics_flag(x):
    x = str(x)
    if "Intel" in x or "UHD" in x or "Iris" in x:
        return 0
    return 1

df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

# ---------------- KNN MODEL FOR RECOMMEND ----------------
X = df[["Price","Ram_GB","SSD_GB","Rating","Graphics_Flag"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn.fit(X_scaled)

# ---------------- FUNCTIONS ----------------

# --- Recommend Tab ---
def recommend(user_type,budget,ram,ssd,rating,graphics):
    if user_type=="Student":
        budget = min(budget,60000)
    elif user_type=="Gamer":
        ram = max(ram,16)
    elif user_type=="Office":
        rating = max(rating,3.5)
    
    g_flag = 0 if graphics=="Integrated" else 1
    user_input = [[budget,ram,ssd,rating,g_flag]]
    user_scaled = scaler.transform(user_input)
    
    distances,indices = knn.kneighbors(user_scaled)
    
    output="# 🎯 Recommended Laptops\n\n"
    
    for i,idx in enumerate(indices[0],1):
        row = df_display.iloc[idx]
        match = max(0,100 - distances[0][i-1]*10)
        query = urllib.parse.quote(str(row.Model))
        amazon_link = f"https://www.amazon.in/s?k={query}"
        
        output+=f"## {i}. {row.Model} ({match:.1f}% match)\n"
        output+=f"- 💰 Price: {row.Price}\n"
        output+=f"- 💾 RAM: {row.Ram}\n"
        output+=f"- 💿 SSD: {row.SSD}\n"
        output+=f"- 🎮 Graphics: {row.Graphics}\n"
        output+=f"- ⭐ Rating: {row.Rating}\n"
        output+=f"- 🛒 [Buy on Amazon]({amazon_link})\n\n"
        output+=f"**Explainable AI:** Recommended based on your {user_type} profile, budget, RAM, and graphics preference.\n\n"
        
    return output

# --- Search Tab ---
def search_laptop(query):
    if not query:
        return "Please enter a search term."
    
    filtered = df_display[df_display["Model"].str.contains(query,case=False,na=False)]
    if filtered.empty:
        return f"No results found for '{query}'."
    
    output=f"# 🔍 Search Results ({len(filtered)})\n\n"
    for i,row in enumerate(filtered.head(10).itertuples(),1):
        query_amz = urllib.parse.quote(str(row.Model))
        amazon_link = f"https://www.amazon.in/s?k={query_amz}"
        output+=f"### {i}. {row.Model}\n- 💰 Price: {row.Price}\n- 💾 RAM: {row.Ram}\n- 💿 SSD: {row.SSD}\n- 🎮 Graphics: {row.Graphics}\n- ⭐ Rating: {row.Rating}\n- 🛒 [Buy on Amazon]({amazon_link})\n\n"
    return output

# --- Price Filter Tab ---
def filter_price(min_price, max_price):
    # Ensure Price numeric
    df_display["Price"] = pd.to_numeric(df_display["Price"], errors="coerce")
    
    filtered = df_display[(df_display["Price"] >= min_price) & (df_display["Price"] <= max_price)]
    if filtered.empty:
        return f"❌ No laptops found in price range ₹{min_price}-₹{max_price}"
    
    filtered = filtered.sort_values(by=["Rating","Price"], ascending=[False, True]).head(50)
    
    output=f"# 💰 Laptops in ₹{min_price}-₹{max_price}\n\n"
    for i, row in enumerate(filtered.itertuples(),1):
        query_amz = urllib.parse.quote(str(row.Model))
        amazon_link = f"https://www.amazon.in/s?k={query_amz}"
        output+=f"### {i}. {row.Model}\n- 💰 Price: {row.Price}\n- 💾 RAM: {row.Ram}\n- 💿 SSD: {row.SSD}\n- 🎮 Graphics: {row.Graphics}\n- ⭐ Rating: {row.Rating}\n- 🛒 [Buy on Amazon]({amazon_link})\n\n"
    return output

# --- Laptop Use Case Advisor Tab ---
def laptop_use_case(use_case):
    if use_case=="Gaming":
        filtered = df_display[(df_display["Graphics_Flag"]==1) & (df_display["Ram_GB"]>=16) & (df_display["SSD_GB"]>=512)]
    elif use_case=="Programming / Development":
        filtered = df_display[(df_display["Ram_GB"]>=8) & (df_display["SSD_GB"]>=256)]
    elif use_case=="Video Editing / Graphics":
        filtered = df_display[(df_display["Ram_GB"]>=16) & (df_display["SSD_GB"]>=512) & (df_display["Graphics_Flag"]==1)]
    elif use_case=="Office / Students":
        filtered = df_display[(df_display["Rating"]>=3.5)]
    else:
        filtered = df_display.copy()
    
    filtered = filtered.sort_values(by=["Rating","Price"], ascending=[False, True]).head(50)
    
    output=f"# 💡 Top Laptops for {use_case}\n\n"
    for i,row in enumerate(filtered.head(5).itertuples(),1):
        query_amz = urllib.parse.quote(str(row.Model))
        amazon_link = f"https://www.amazon.in/s?k={query_amz}"
        output+=f"### {i}. {row.Model}\n- 💰 Price: {row.Price}\n- 💾 RAM: {row.Ram}\n- 💿 SSD: {row.SSD}\n- 🎮 Graphics: {row.Graphics}\n- ⭐ Rating: {row.Rating}\n- 🛒 [Buy on Amazon]({amazon_link})\n"
        output+=f"**Explainable AI:** Recommended because it matches your use case requirements.\n\n"
    return output

# ---------------- UI ----------------
with gr.Blocks(title="Laptop Finder AI") as app:
    gr.Markdown("# 💻 Laptop Finder AI\n### Smart Laptop Recommendation System with Explainable AI")
    
    with gr.Tabs():
        
        # --- Recommend Tab ---
        with gr.Tab("🔍 Recommend"):
            user_type = gr.Radio(["Student","Office","Gamer"],value="Student",label="User Type")
            budget = gr.Slider(20000,200000,60000,step=5000,label="Budget (₹)")
            ram = gr.Dropdown([4,8,16,32],value=8,label="RAM (GB)")
            ssd = gr.Dropdown([256,512,1024],value=512,label="SSD (GB)")
            rating = gr.Slider(0,5,3,step=0.1,label="Minimum Rating")
            graphics = gr.Radio(["Integrated","Dedicated"],value="Dedicated")
            
            btn = gr.Button("Find Best Laptop",variant="primary")
            output = gr.Markdown()
            btn.click(recommend,inputs=[user_type,budget,ram,ssd,rating,graphics],outputs=output)
        
        # --- Search Tab ---
        with gr.Tab("🔎 Search"):
            search_input = gr.Textbox(label="Search Laptop (Brand/Model)",placeholder="HP, Lenovo...")
            search_btn = gr.Button("Search")
            search_output = gr.Markdown()
            search_btn.click(search_laptop,search_input,search_output)
        
       
       # --- Price Filter Tab ---
with gr.Tab("💰 Price Filter"):
    min_price_slider = gr.Slider(
        minimum=int(df_display["Price"].min()),
        maximum=int(df_display["Price"].max()),
        value=int(df_display["Price"].min()),
        step=5000,
        label="Minimum Price (₹)"
    )

    max_price_slider = gr.Slider(
        minimum=int(df_display["Price"].min()),
        maximum=int(df_display["Price"].max()),
        value=int(df_display["Price"].max()),
        step=5000,
        label="Maximum Price (₹)"
    )

    price_btn = gr.Button("Filter Laptops")
    price_output = gr.Markdown()

    price_btn.click(
        filter_price,
        inputs=[min_price_slider, max_price_slider],
        outputs=price_output
    )

        
        # --- Laptop Use Case Advisor Tab ---
        with gr.Tab("💡 Laptop Use Case Advisor"):
            use_case_dropdown = gr.Dropdown(["Gaming","Programming / Development","Video Editing / Graphics","Office / Students"],value="Office / Students",label="Select Use Case")
            use_case_btn = gr.Button("Show Laptops")
            use_case_output = gr.Markdown()
            use_case_btn.click(laptop_use_case,use_case_dropdown,use_case_output)

# ---------------- LAUNCH ----------------
app.launch()
