# app.py
# Web demo: Phân tích giá trị bất thường trong báo cáo tài chính
# Framework: Streamlit
# AI Model API: Groq
#
# Cài thư viện:
# pip install streamlit pandas openpyxl groq matplotlib seaborn
#
# Chạy:
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import matplotlib.pyplot as plt

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Financial Statement Anomaly Detection",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Financial Statement Anomaly Detection")
st.markdown(
    """
    Demo phát hiện giá trị bất thường trong báo cáo tài chính bằng AI + thống kê.
    """
)

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Cấu hình")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Chọn model",
    [
        "llama-3.3-70b-versatile",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]
)

threshold = st.sidebar.slider(
    "Ngưỡng Z-score",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.1
)

uploaded_file = st.file_uploader(
    "📂 Upload file Excel hoặc CSV",
    type=["csv", "xlsx"]
)

# =========================
# ĐỌC FILE
# =========================
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    return df


# =========================
# PHÁT HIỆN BẤT THƯỜNG
# =========================
def detect_anomalies(df, threshold=2.5):
    numeric_cols = df.select_dtypes(include=np.number).columns

    anomaly_rows = []

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        if std == 0:
            continue

        z_scores = (df[col] - mean) / std

        anomalies = df[np.abs(z_scores) > threshold].copy()

        anomalies["Anomaly_Column"] = col
        anomalies["Z_Score"] = z_scores[np.abs(z_scores) > threshold]

        anomaly_rows.append(anomalies)

    if anomaly_rows:
        result = pd.concat(anomaly_rows)
    else:
        result = pd.DataFrame()

    return result


# =========================
# AI PHÂN TÍCH
# =========================
def analyze_with_ai(api_key, model, anomalies_df):
    client = Groq(api_key=api_key)

    preview = anomalies_df.head(20).to_string()

    prompt = f"""
Bạn là chuyên gia phân tích tài chính.

Dưới đây là các giá trị bất thường được phát hiện trong báo cáo tài chính:

{preview}

Hãy:
1. Giải thích các bất thường.
2. Đưa ra khả năng gian lận hoặc sai sót.
3. Đề xuất hướng kiểm tra.
4. Đánh giá mức độ rủi ro.

Trả lời bằng tiếng Việt rõ ràng.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content


# =========================
# MAIN
# =========================
if uploaded_file:

    df = load_data(uploaded_file)

    st.subheader("📄 Dữ liệu đầu vào")

    st.dataframe(df, use_container_width=True)

    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    anomalies_df = detect_anomalies(df, threshold)

    st.subheader("🚨 Các giá trị bất thường")

    if anomalies_df.empty:
        st.success("Không phát hiện giá trị bất thường.")
    else:
        st.dataframe(anomalies_df, use_container_width=True)

        # =========================
        # BIỂU ĐỒ
        # =========================
        st.subheader("📈 Visualization")

        numeric_cols = df.select_dtypes(include=np.number).columns

        selected_col = st.selectbox(
            "Chọn cột để hiển thị",
            numeric_cols
        )

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(df[selected_col], marker='o')

        mean = df[selected_col].mean()
        std = df[selected_col].std()

        upper = mean + threshold * std
        lower = mean - threshold * std

        ax.axhline(upper, linestyle='--')
        ax.axhline(lower, linestyle='--')

        ax.set_title(f"Anomaly Detection - {selected_col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(selected_col)

        st.pyplot(fig)

        # =========================
        # AI ANALYSIS
        # =========================
        st.subheader("🤖 AI Financial Analysis")

        if not groq_api_key:
            st.warning("Vui lòng nhập Groq API Key.")
        else:
            if st.button("Phân tích bằng AI"):

                with st.spinner("AI đang phân tích..."):

                    try:
                        ai_result = analyze_with_ai(
                            groq_api_key,
                            model_name,
                            anomalies_df
                        )

                        st.markdown(ai_result)

                    except Exception as e:
                        st.error(f"Lỗi API: {e}")

else:
    st.info("Hãy upload file báo cáo tài chính để bắt đầu.")