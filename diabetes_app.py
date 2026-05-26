import streamlit as st
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 예측 시스템",
    page_icon="🏥",
    layout="centered"
)

# 병원 스타일 CSS
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background-color: #f5f7fa;
}

/* 제목 */
h1 {
    color: #1565c0;
    text-align: center;
    font-size: 2.7rem !important;
    margin-bottom: 10px;
}

/* 설명 */
.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}

/* 카드 UI */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.form-card {
    background-color: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* 입력창 */
[data-testid="stNumberInput"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 8px;
}

/* 버튼 */
.stButton>button {
    width: 100%;
    background-color: #1976d2;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 18px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1565c0;
    color: white;
}

/* 결과 박스 */
.result-box {
    background-color: white;
    border-radius: 18px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.result-title {
    color: #1565c0;
    font-size: 24px;
    font-weight: bold;
}

.result-value {
    font-size: 36px;
    font-weight: bold;
    margin-top: 15px;
    margin-bottom: 10px;
}

.result-prob {
    font-size: 20px;
    color: #444;
}

</style>
""", unsafe_allow_html=True)

# 모델 불러오기
model = joblib.load("diabetes_model.pkl")

# 제목
st.markdown("<h1>🏥 당뇨병 예측 시스템</h1>", unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">환자 정보를 입력하여 당뇨 위험도를 예측합니다.</div>',
    unsafe_allow_html=True
)

# 입력 영역
st.markdown('<div class="form-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("임신횟수", value=0.0)
    glucose = st.number_input("포도당(혈당)", value=120.0)
    bp = st.number_input("혈압", value=70.0)
    skin = st.number_input("피부 두께", value=20.0)

with col2:
    insulin = st.number_input("인슐린", value=80.0)
    bmi = st.number_input("체질량지수(BMI)", value=25.0)
    dpf = st.number_input("당뇨내력가중치", value=0.5)
    age = st.number_input("나이", value=30.0)

st.markdown("</div>", unsafe_allow_html=True)

# 예측 버튼
if st.button("예측하기"):

    # 입력 데이터 생성
    input_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            '임신횟수',
            '포도당(혈당)',
            '혈압',
            '피부 두께',
            '인슐린',
            '체질량지수',
            '당뇨내력가중치',
            '나이'
        ]
    )

    # 예측
    predicted = model.predict(input_data)

    # 확률 계산
    try:
        prob = model.predict_proba(input_data)
        probability = prob[0][1] * 100
        probability_text = f"{probability:.1f}%"
    except:
        probability_text = "확인 불가"

    # 결과 출력
    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">예측 결과</div>

            <div class="result-value">
                {"당뇨 위험" if predicted[0] == 1 else "정상 범위"}
            </div>

            <div class="result-prob">
                당뇨 확률 : <b>{probability_text}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 추가 메시지
    if predicted[0] == 1:
        st.error("의료진 상담 및 생활습관 관리가 권장됩니다.")
    else:
        st.success("현재는 비교적 안정적인 상태입니다.")
