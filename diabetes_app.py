import streamlit as st
import pandas as pd
import joblib

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

/* 입력칸 */
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

.stButton>button:hover {
    background-color: #b30000;
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

# 모델 불러오기
model = joblib.load("diabetes_model.pkl")

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

    # predict_proba 가능한 경우만
    try:
        prob = model.predict_proba(input_data)
        probability_text = f"{prob[0][1] * 100:.1f}%"
    except:
        probability_text = "확인 불가"

    # 결과 출력
    st.markdown(
        f'''
        <div class="result-box">
            ☠️ 예측 결과 ☠️<br><br>
            판정 : <b>{predicted[0]}</b><br><br>
            🩸 당뇨 확률 : <b>{probability_text}</b>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # 결과 메시지
    if predicted[0] == 0:
        st.success("😌 아직은 안전합니다...")
    else:
        st.error("🚨 위험 신호 감지... 건강 관리가 필요합니다!")
