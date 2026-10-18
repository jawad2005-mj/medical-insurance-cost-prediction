import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥")


@st.cache_resource
def load_bundle():
    return joblib.load("insurance_model.joblib")


def predict_cost(age, bmi, smoker, children, sex, region):
    bundle = load_bundle()
    enc = bundle["encodings"]
    row = pd.DataFrame(
        [[age, enc["sex"][sex], bmi, children, enc["smoker"][smoker], enc["region"][region]]],
        columns=bundle["features"],
    )
    # Linear model negative de sakta hai (young, healthy, low BMI) — cost negative nahi ho sakti
    return max(float(bundle["model"].predict(row)[0]), 0.0)


st.title("🏥 Medical Insurance Cost Predictor")
st.caption("Linear Regression model (Medical Cost Personal dataset) par based estimate.")

with st.form("inputs"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    bmi = c2.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    sex = c1.selectbox("Gender", ["female", "male"])
    smoker = c2.selectbox("Smoker?", ["no", "yes"])
    children = c1.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
    region = c2.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    go = st.form_submit_button("Predict cost", use_container_width=True)

if go:
    cost = predict_cost(age, bmi, smoker, children, sex, region)
    st.metric("Estimated yearly insurance cost", f"${cost:,.2f}")
    st.caption("Yeh sirf ek estimate hai; model ka test R² ≈ 0.81 aur average error ≈ $4,200 hai.")
