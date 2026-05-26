import streamlit as st
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(
    page_title="당뇨 예측 시스템",
    page_icon="🩸",
    layout="centered"
)

# 호러 영화 스타일 CSS
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background:
        radial-gradient(circle at top left, rgba(120,0,0,0.25), transparent 25%),
        radial-gradient(circle at bottom right, rgba(255,0,0,0.15), transparent 20%),
        linear-gradient(to bottom, #050505, #111111);
    color: white;
    overflow-x: hidden;
}

/* 핏자국 효과 */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 180px;
    background:
        radial-gradient(circle, rgba(180,0,0,0.9) 10%, transparent 11%),
        radial-gradient(circle, rgba(120,0,0,0.8) 8%, transparent 9%),
        radial-gradient(circle, rgba(255,0,0,0.7) 6%, transparent 7%);
    background-size: 120px 120px;
    opacity: 0.4;
    pointer-events: none;
    filter: blur(2px);
}

/* 제목 */
h1 {
    color: #ff3b3b;
    text-align: center;
    font-size: 3rem !important;
    letter-spacing: 2px;
    text-shadow:
        0px 0px 10px #ff0000,
        0px 0px 25px #8b0000;
    margin-bottom: 10px;
}

/* 텍스트 */
p, label, div {
    color: #f1f1f1 !important;
}

/* 입력창 */
[data-testid="stNumberInput"] {
    background-color: rgba(20,20,20,0.95);
    padding: 10px;
    border-radius: 15px;
    border: 2px solid #5a0000;
    box-shadow: 0px 0px 10px rgba(255,0,0,0.2);
}

/* 입력칸 */
input {
    color: white !important;
    background-color: #111111 !important;
}

/* 버튼 */
.stButton>button {
    background: linear-gradient(to bottom, #7a0000, #3a0000);
    color: white;
    border-radius: 15px;
    border: 2px solid #ff4d4d;
    padding: 0.9rem 2.2rem;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 1px;
    box-shadow:
        0px 0px 12px rgba(255,0,0,0.5),
        inset 0px 0px 10px rgba(255,255,255,0.1);
}

/* 버튼 호버 */
.stButton>button:hover {
    background: linear-gradient(to bottom, #a30000, #520000);
    transform: scale(1.02);
}

/* 결과 박스 */
.result-box {
    background: rgba(10,10,10,0.95);
    padding: 25px;
    border-radius: 20px;
    border: 2px solid #ff2e2e;
    text-align: center;
    font-size: 28px;
    color: #ff4d4d;
    margin-top: 30px;
    box-shadow:
        0px 0px 20px rgba(255,0,0,0.5),
        inset 0px 0px 15px rgba(255,0,0,0.2);
}

/* 스크롤바 */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #7a0000;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# 모델 불러오기
model = joblib.load("diabetes_model.pkl")

# 제목
st.markdown(
    "<h1>🩸 당뇨 예측 시스템</h1>",
    unsafe_allow_html=True
)

st.write("### 건강 상태를 입력하세요...")

# 사용자 입력
preg = st.number_input("🤰 임신횟수", value=0.0)
glucose = st.number_input("🩸 포도당(혈당)", value=120.0)
bp = st.number_input("💓 혈압", value=70.0)
skin = st.number_input("🖐 피부 두께", value=20.0)
insulin = st.number_input("💉 인슐린", value=80.0)
bmi = st.number_input("⚖️ 체질량지수(BMI)", value=25.0)
dpf = st.number_input("🧬 당뇨내력가중치", value=0.5)
age = st.number_input("🎂 나이", value=30.0)

# 버튼
if st.button("예측 시작"):

    # 데이터 생성
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

    # 확률 계산 시도
    try:
        prob = model.predict_proba(input_data)
        probability_text = f"{prob[0][1] * 100:.1f}%"
    except:
        probability_text = "확인 불가"

    # 결과 출력
    st.markdown(
        f'''
        <div class="result-box">
            🩸 예측 결과 🩸<br><br>
            판정 : <b>{predicted[0]}</b><br><br>
            당뇨 확률 : <b>{probability_text}</b>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # 결과 메시지
    if predicted[0] == 0:
        st.success("현재는 비교적 안정적인 상태입니다.")
    else:
        st.error("위험 신호가 감지되었습니다.")
