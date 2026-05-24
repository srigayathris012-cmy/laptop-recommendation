# 💻 LaptopAI — Intelligent Laptop Recommendation System

> An AI-powered web application that recommends laptops using Machine Learning
> and answers user queries through a Claude AI chatbot assistant.

---

## 🚀 Live Demo

👉 **[Try the App Here](https://laptop-recommendation-cuvcqnxtj2nzjhutlmeuds.streamlit.app/)**

![LaptopAI App Screenshot](<img width="1568" height="742" alt="screenshot" src="https://github.com/user-attachments/assets/bfd131b2-31ec-4715-988e-8a2a3ab3847e" />
)

---

## ✨ Features

| Tab | Description |
|-----|-------------|
| 🔍 Smart Recommend | KNN-based ML recommendations from user preferences |
| 🤖 AI Assistant | Claude API powered chatbot in sidebar |
| 🔎 Search | Search laptops by brand or model name |
| 💰 Price Filter | Dynamic price range filtering |
| 🧠 Deep Insights | Usage fit scores, value score, future-proof analysis |
| 📈 Trending | Algorithm-based trending score ranking |
| 🛒 Amazon Links | Direct buy links for every laptop |

---

## 🛠️ Tech Stack

```
Python        → Core programming language
Scikit-learn  → KNN algorithm + StandardScaler
Streamlit     → Web app framework and deployment
Pandas        → Data loading and cleaning
NumPy         → Numerical operations
aiohttp       → Async Claude API calls
urllib        → Amazon link generation
```

---

## 🧠 How the ML Works

```
User Input (Budget, RAM, SSD, Rating, Graphics)
          ↓
   StandardScaler (normalize all features equally)
          ↓
   KNN Algorithm (find 5 nearest laptops)
          ↓
   Distance Score → Match Percentage
          ↓
   Top 5 Recommendations displayed
```

**Why StandardScaler?**
Price ranges ₹20,000–₹2,00,000 while RAM ranges 4–32GB.
Without scaling, price dominates distance calculation and
RAM/SSD get ignored. StandardScaler makes all features equal weight.

**Why KNN?**
KNN finds laptops most similar to the user's requirement
vector by calculating Euclidean distance across all features.
n_neighbors=5 returns the 5 closest matching laptops.

---

## 🤖 AI Chatbot

The sidebar AI Assistant uses the **Claude API (claude-sonnet-4)**
via async HTTP calls. It receives real laptop data as context
and answers natural language questions like:
- "Best gaming laptop under 60k"
- "Laptop for students"
- "Best value for money"

If the API call fails, a smart rule-based fallback activates automatically.

---

## 📊 Dataset

- 920+ laptops with real specs and prices
- Features: Brand, Model, RAM, SSD, Graphics, Price, Rating
- Custom data cleaning: regex price parser, GPU flag logic
- Source: Collected from online laptop listings

---

## 🚀 How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/laptop-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
# Create .streamlit/secrets.toml and add:
# ANTHROPIC_API_KEY = "your-key-here"

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
laptop-ai/
│
├── app.py              ← Main Streamlit application
├── laptop.csv          ← Dataset (920+ laptops)
├── requirements.txt    ← Python dependencies
├── screenshot.png      ← App screenshot
└── README.md           ← This file
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
aiohttp
```

---

## 🎯 Scoring System

**Match %** → Based on KNN distance score (closer = higher match)

**Trending Score** → Rating (50%) + Value Score (30%) + Specs bonus (20%)

**Future-proof Score** → RAM≥16GB (+40) + SSD≥512GB (+30) + Dedicated GPU (+20) + Rating≥4 (+10)

**Value Score** → (RAM×2 + SSD/256 + Rating×5) / Price

---

## 👩‍💻 Author

**Sri Gayathri S** — BE CSE, Kamaraj College of Engineering and Technology

- 🎓 Internship: Panith Innovations, Bangalore (ML & AI Deployment)
- 📅 Built during: 15-day AI/ML Internship (Dec 2025)

---

## 📜 Internship Certificate

This project was developed as the final project during a
**15-day AI/ML Internship** at Panith Innovations, Bangalore
in association with Paruvaththe Payir Sei (Dec 2025).

---


