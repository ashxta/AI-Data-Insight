import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# ==== Hugging Face API====
HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta" 
HF_TOKEN = "my-token"  

def generate_ai_insights(prompt):
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
    
def clean_data(df):
    df = df.dropna(how='all')
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def plot_visualizations(df):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        st.subheader("📊 Data Visualizations")
        for col in numeric_cols:
            st.write(f"Histogram for **{col}**")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
        if len(numeric_cols) > 1:
            st.write("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    else:
        st.info("No numeric columns to visualize.")

def main():
    st.set_page_config(page_title="📊 Free AI Data Analyzer", layout="wide")
    st.title("🤖 AI-Powered Data Insights")

    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Original Data")
        st.dataframe(df.head())

        df_clean = clean_data(df)
        st.subheader("🧼 Cleaned Data")
        st.dataframe(df_clean.head())

        plot_visualizations(df_clean)

        with st.spinner("Generating AI insights (Free Zephyr-7B)..."):
            summary = df_clean.describe().transpose().to_string()
            insights = generate_ai_insights(summary)

        st.subheader("🧠 AI-Generated Insights (Zephyr 7B)")
        st.markdown(insights)

if __name__ == "__main__":
    main()