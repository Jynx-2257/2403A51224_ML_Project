import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# --- CHESS.COM ULTIMATE THEME ---
st.set_page_config(page_title="Chess Analysis - Checkmate Catalyst", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #312e2b; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #262421 !important; width: 300px !important; }
    
    /* Sidebar text and inputs */
    section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label {
        color: #bababa !important; font-family: "Segoe UI", Arial, sans-serif;
    }

    /* Evaluation Bar Mimicry */
    .eval-bar-container {
        width: 30px; height: 400px; background-color: #403d39;
        border-radius: 4px; position: relative; margin-right: 20px;
    }
    .eval-bar-fill {
        width: 100%; position: absolute; bottom: 0;
        background-color: #ffffff; transition: height 0.5s ease-in-out;
        border-radius: 0 0 4px 4px;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background-color: #262421; border: 1px solid #403d39;
        border-radius: 8px; padding: 15px;
    }

    /* Chess.com Button */
    .stButton>button {
        background-color: #81b64c; color: white; border-radius: 8px;
        font-weight: bold; border: none; height: 3em; width: 100%;
        border-bottom: 4px solid #457531;
    }
    .stButton>button:hover { background-color: #a3d16a; color: white; }
    
    /* Board Coordinates labels */
    .coord { color: #7d7a77; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def load_data():
    chess = fetch_openml(data_id=3, as_frame=True, parser='auto')
    df = chess.frame
    X = pd.get_dummies(df.drop('class', axis=1))
    y = df['class'].apply(lambda x: 1 if x == 'won' else 0)
    return X, y, list(X.columns)

X, y, feature_names = load_data()

# --- SIDEBAR (The Control Panel) ---
with st.sidebar:
    st.image("https://www.chess.com/bundles/web/images/user-setup/monospaced-dark.81b64c.svg", width=150)
    st.markdown("### **Analysis Settings**")
    depth = st.slider("Engine Depth", 1, 20, 7)
    crit = st.selectbox("Heuristic Mode", ["Gini", "Entropy"])
    st.divider()
    st.markdown("### **Unit III Concept**")
    st.caption("Topic 25: Information Gain via recursive binary splitting.")

# --- MODEL TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(max_depth=depth, criterion=crit.lower())
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)

# --- MAIN INTERFACE (Analysis Board Layout) ---
col_eval, col_board, col_moves = st.columns([0.1, 1.2, 0.8])

with col_eval:
    # The vertical evaluation bar
    eval_height = int(acc * 100)
    st.markdown(f"""
        <div class="eval-bar-container">
            <div class="eval-bar-fill" style="height: {eval_height}%;"></div>
        </div>
        <p style='text-align: center; color: white; font-size: 10px; margin-top: 5px;'>{acc:.2f}</p>
        """, unsafe_allow_html=True)

with col_board:
    st.markdown("<h3 style='margin-bottom: 0px;'>Analysis Board</h3>", unsafe_allow_html=True)
    st.caption("Visualizing Decision Tree splits as strategic board logic.")
    
    # Modernized Tree Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#312e2b')
    plot_tree(clf, max_depth=2, feature_names=feature_names, 
              class_names=['Draw', 'Win'], filled=True, rounded=True,
              precision=1, fontsize=9)
    
    # Customizing plot colors to match theme
    for text in ax.texts:
        text.set_color('black')
    st.pyplot(fig)

with col_moves:
    st.markdown("### **Game Review**")
    
    st.metric("Accuracy", f"{acc:.1%}", delta="Great Move")
    
    st.markdown("---")
    st.write("**Top Informative Squares**")
    
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(6)
    for feat, val in importances.items():
        clean_name = feat.replace('_', ' ').title()
        st.write(f"**{clean_name}**")
        st.progress(float(val / importances.max()))
        
    st.markdown("---")
    if st.button("RUN ENGINE ANALYSIS"):
        st.balloons()

# --- FOOTER ---
st.markdown("<br><p style='text-align: center; color: #7d7a77;'>Chess.com Catalyst Interface | Unit III Project</p>", unsafe_allow_html=True)

# Type "streamlit run app5.py" after a series of lines are shown in the terminal without any errors to run the file.
