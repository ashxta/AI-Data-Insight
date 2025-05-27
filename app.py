import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from datetime import datetime

# ====== 🎨 CUSTOM STYLING ======
st.set_page_config(
    page_title="📊 AI Data Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .sidebar .sidebar-content { background-color: #2c3e50; color: white; }
    h1, h2, h3 { color: #2c3e50 !important; }
    .stButton button { background-color: #3498db !important; color: white !important; }
    .stProgress > div > div > div { background-color: #3498db; }
    .stDataFrame { font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# ====== 🔧 FUNCTIONS ======
@st.cache_data
def clean_data(df):
    """Clean the dataframe by handling missing values and converting data types"""
    # Drop completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Convert potential date columns
    date_cols = ['date_added']  # Add other date columns as needed
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert duration to numeric if it exists
    if 'duration' in df.columns:
        df['duration_numeric'] = pd.to_numeric(df['duration'].str.extract('(\d+)')[0], errors='coerce')
    
    # Fill missing values
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Strip and standardize text columns (categorical cleanup)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.capitalize()
    
    return df


def generate_ai_insights(prompt):
    HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    HF_TOKEN = "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"  # Replace with your token
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": f"""<|system|>You are a data analyst. Provide 3-5 concise insights in bullet points.</|system|>
        <|user|>Analyze this data summary:\n{prompt}</|user|>
        <|assistant|>""",
        "parameters": {"max_new_tokens": 250}
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text'].split("<|assistant|>")[-1].strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ====== 🖥️ MAIN APP ======
def main():
    st.title("📊 AI Data Analyzer")
    st.markdown("Upload a CSV file to get **automated data cleaning, visualizations, and AI insights**.")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("This app uses **Zephyr-7B** (AI) for insights.")
        st.markdown("### How to Use:")
        st.markdown("1. **Upload** a CSV file\n2. **Explore** insights & visualizations\n3. **Custom charting available!**")
        st.markdown("---")

    uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = pd.read_csv(uploaded_file)
            time.sleep(1)

        st.subheader("🔍 Original Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        st.subheader("🧼 Data Cleaning")
        with st.spinner("Cleaning data..."):
            df_clean = clean_data(df)
            time.sleep(1)

        st.dataframe(df_clean.head(), use_container_width=True)
        st.write(f"After cleaning: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")

        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        non_numeric_cols = [col for col in df_clean.columns if col not in numeric_cols]

        st.subheader("📋 Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numeric Columns:**")
            st.write(numeric_cols if numeric_cols else "No numeric columns found")
        with col2:
            st.markdown("**Non-Numeric Columns:**")
            st.write(non_numeric_cols if non_numeric_cols else "All columns are numeric")

        # === 📊 VISUALIZATION SECTION ===
        st.subheader("📊 Interactive Visualizations")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Chart by Column", "Correlations", "Scatter Plots", "XY Plot Builder"
        ])

        # TAB 1: Custom chart by column
        with tab1:
            st.markdown("### Custom Column Visualization")
            chart_type = st.selectbox("Choose chart type:", [
                "Histogram", "Box Plot", "Bar Chart (Categorical)", "Pie Chart (Categorical)", "Line Chart"
            ])

            all_cols = df_clean.columns.tolist()
            if chart_type in ["Histogram", "Box Plot", "Line Chart"]:
                col = st.selectbox("Select numeric column:", numeric_cols)
            else:
                col = st.selectbox("Select categorical column:", non_numeric_cols)

            fig, ax = plt.subplots(figsize=(8, 5))
            if chart_type == "Histogram":
                sns.histplot(df_clean[col], kde=True, ax=ax)
                ax.set_title(f'Histogram of {col}')
            elif chart_type == "Box Plot":
                sns.boxplot(x=df_clean[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
            elif chart_type == "Bar Chart (Categorical)":
                top_cat = df_clean[col].value_counts().nlargest(10)
                sns.barplot(x=top_cat.values, y=top_cat.index, ax=ax)
                ax.set_title(f'Bar Chart of {col}')
            elif chart_type == "Pie Chart (Categorical)":
                top_cat = df_clean[col].value_counts().nlargest(5)
                fig, ax = plt.subplots()
                ax.pie(top_cat.values, labels=top_cat.index, autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Pie Chart of {col}')
            elif chart_type == "Line Chart":
                ax.plot(df_clean[col].dropna().values)
                ax.set_title(f'Line Chart of {col}')
                ax.set_ylabel(col)
                ax.set_xlabel("Index")
            st.pyplot(fig)

        # TAB 2: Correlations
        with tab2:
            st.markdown("### Correlation Analysis")
            if len(numeric_cols) > 1:
                corr = df_clean[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title("Correlation Heatmap")
                st.pyplot(fig)

                upper = corr.abs().where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                top_corr = upper.stack().sort_values(ascending=False).head(5)
                st.markdown("**Top Correlations:**")
                st.write(top_corr)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")

        # TAB 3: Scatter Plot
        with tab3:
            st.markdown("### Scatter Plot")
            if len(numeric_cols) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols, index=0)
                with col2:
                    y_col = st.selectbox("Y-axis:", numeric_cols, index=1)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.regplot(data=df_clean, x=x_col, y=y_col, ax=ax, scatter_kws={'alpha': 0.5})
                ax.set_title(f'{x_col} vs {y_col}')
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns.")

        # TAB 4: XY Plot Builder
        with tab4:
            st.markdown("### XY Chart Builder (Flexible)")
            x_axis = st.selectbox("Choose X-axis column:", df_clean.columns)
            y_axis = st.selectbox("Choose Y-axis column:", df_clean.columns)
            plot_kind = st.selectbox("Chart type:", ["Scatter", "Line", "Bar"])

            fig, ax = plt.subplots(figsize=(8, 6))
            if plot_kind == "Scatter":
                ax.scatter(df_clean[x_axis], df_clean[y_axis], alpha=0.6)
            elif plot_kind == "Line":
                ax.plot(df_clean[x_axis], df_clean[y_axis])
            elif plot_kind == "Bar":
                ax.bar(df_clean[x_axis], df_clean[y_axis])
            ax.set_title(f'{plot_kind} plot of {x_axis} vs {y_axis}')
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # AI INSIGHTS
        st.subheader("🧠 AI-Generated Insights")
        with st.spinner("Generating insights ..."):
            summary = f"""
            Data Overview:
            - Total rows: {df_clean.shape[0]}
            - Total columns: {df_clean.shape[1]}
            - Numeric columns: {len(numeric_cols)}
            - Non-numeric columns: {len(non_numeric_cols)}

            Summary Statistics:
            {df_clean.describe().to_string()}
            """
            insights = generate_ai_insights(summary)
            time.sleep(2)
        st.success("✅ Analysis Complete!")
        st.markdown(insights)

if __name__ == "__main__":
    main()
