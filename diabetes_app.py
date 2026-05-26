import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="공포의 당뇨 예측 시스템",
    page_icon="💀",
    layout="centered"
)

# 공포 UI CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom, #050505, #111111);
    color: white;
}

/* 제목 */
h1 {
    color: #ff2e2e;
    text-align: center;
    font-size: 3rem !important;
    text-shadow: 0px 0px 15px red;
}

/* 텍스트 */
p, label, div {
    color: #f1f1f1 !important;
}

/* 입력창 */
[data-testid="stNumberInput"] {
    background-color: #1a1a1a;
    padding: 10px;
    border-radius: 15px;
    border: 2px solid #8b0000;
}

/* 입력칸 내부 */
input {
    color: white !important;
    background-color: #111111 !important;
}

/* 버튼 */
.stButton>button {
    background-color: #8b0000;
    color: white;
    border-radius: 15px;
    border: 2px solid #ff4d4d;
    padding: 0.8rem 2rem;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0px 0px 15px red;
}

/* 버튼 호버 */
.stButton>button:hover {
    background-color: #b30000;
    color: white;
}

/* 결과 박스 */
.result-box {
    background-color: #111111;
    padding: 25px;
    border-radius: 20px;
    border: 3px solid #ff2e2e;
    text-align: center;
    font-size: 28px;
    color: #ff4d4d;
    margin-top: 25px;
    box-shadow: 0px 0px 20px red;
}

</style>
""", unsafe_allow_html=True)

# 모델 & 스케일러 불러오기
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# 제목
st.markdown(
    "<h1>💀 공포의 당뇨 예측 시스템 🩸</h1>",
    unsafe_allow_html=True
)

st.write("### ☠️ 당신의 건강 상태를 확인해보세요...")

# 사용자 입력
preg = st.number_input("🤰 임신횟수", value=0.0)
glucose = st.number_input("🩸 포도당(혈당)", value=120.0)
bp = st.number_input("💓 혈압", value=70.0)
skin = st.number_input("🖐 피부 두께", value=20.0)
insulin = st.number_input("💉 인슐린", value=80.0)
bmi = st.number_input("⚖️ 체질량지수(BMI)", value=25.0)
dpf = st.number_input("🧬 당뇨내력가중치", value=0.5)
age = st.number_input("🎂 나이", value=30.0)

# 예측 버튼
if st.button("🔮 운명을 확인하기"):

    # scaler feature 이름 가져오기
    feature_names = scaler.feature_names_in_

    # 입력값 배열 생성
    values = np.array([
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]).reshape(1, -1)

    # scaler 기준 컬럼명으로 DataFrame 생성
    input_data = pd.DataFrame(values, columns=feature_names)

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    # 결과 출력
    st.markdown(
        f"""
        <div class="result-box">
            ☠️ 예측 결과 ☠️<br><br>
            판정 : <b>{predicted[0]}</b><br><br>
            🩸 당뇨 확률 : <b>{prob[0][1] * 100:.1f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 결과 메시지
    if predicted[0] == 0:
        st.success("😌 아직은 안전합니다...")
    else:
        st.error("🚨 위험 신호 감지... 건강 관리가 필요합니다!")
