import streamlit as st
import pandas as pd
import joblib

# 페이지 설정
st.set_page_config(
    page_title="당뇨병 예측 시스템",
    page_icon="🩺",
    layout="centered"
)

# CSS 꾸미기
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom, #f8fbff, #e3f2fd);
}

h1 {
    color: #1565c0;
    text-align: center;
    font-size: 3rem !important;
}

.stButton>button {
    background-color: #42a5f5;
    color: white;
    border-radius: 15px;
    border: none;
    padding: 0.7rem 2rem;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1e88e5;
    color: white;
}

[data-testid="stNumberInput"] {
    background-color: white;
    padding: 10px;
    border-radius: 15px;
    border: 2px solid #bbdefb;
}

.result-box {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    border: 3px dashed #64b5f6;
    text-align: center;
    font-size: 24px;
    color: #1565c0;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# 모델 & 스케일러 불러오기
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler_final.pkl")

# 제목
st.markdown("<h1>🩺 당뇨병 예측 시스템 💙</h1>", unsafe_allow_html=True)

st.write("### 📋 건강 정보를 입력해주세요!")

# 사용자 입력
preg = st.number_input("🤰 임신횟수", value=0)
glucose = st.number_input("🍬 포도당(혈당)", value=120.0)
bp = st.number_input("💓 혈압", value=70.0)
skin = st.number_input("🖐 피부 두께", value=20.0)
insulin = st.number_input("💉 인슐린", value=80.0)
bmi = st.number_input("⚖️ 체질량지수(BMI)", value=25.0)
dpf = st.number_input("🧬 당뇨내력가중치", value=0.5)
age = st.number_input("🎂 나이", value=30)

# 예측 버튼
if st.button("🔍 당뇨 예측하기"):

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

    # 파생 변수 생성
    input_data['인슐린_포도당_비율'] = input_data['인슐린'] / (input_data['포도당(혈당)'] + 1)

    input_data['신체위험도'] = (
        input_data['체질량지수'] + input_data['혈압']
    )

    input_data['비만혈당지수'] = (
        input_data['체질량지수'] * input_data['포도당(혈당)']
    )

    input_data['임신_나이_비율'] = (
        input_data['임신횟수'] / (input_data['나이'] + 1)
    )

    input_data['건강취약'] = (
        input_data['나이'] * input_data['체질량지수']
    ) / 100

    input_data['고령'] = (
        input_data['나이'] >= 50
    ).astype(int)

    # 컬럼 순서 맞추기
    input_data = input_data[
        [
            '임신횟수',
            '포도당(혈당)',
            '혈압',
            '피부 두께',
            '인슐린',
            '체질량지수',
            '당뇨내력가중치',
            '나이',
            '인슐린_포도당_비율',
            '신체위험도',
            '비만혈당지수',
            '임신_나이_비율',
            '건강취약',
            '고령'
        ]
    ]

    # 스케일링
    input_scaled = scaler.transform(input_data)

    # 예측
    predicted = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    # 결과 출력
    st.markdown(
        f"""
        <div class="result-box">
            🧬 예측 결과 🧬<br><br>
            <b>{predicted[0]}</b><br><br>
            💡 당뇨 확률: <b>{prob[0][1] * 100:.1f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 결과 메시지
    if predicted[0] == 0:
        st.success("💪 현재는 정상 범위로 예측됩니다!")
    else:
        st.error("⚠️ 당뇨 위험 가능성이 있습니다!")
