import streamlit as st
import pickle
import pandas as pd

# Load artifacts
with open('model', 'rb') as f:
    reg = pickle.load(f)

with open('encoder', 'rb') as f:
    oh = pickle.load(f)

with open('scaler', 'rb') as f:
    sc = pickle.load(f)

st.set_page_config(page_title="Startup Profit Predictor", page_icon="🚀")

st.title("🚀 Startup Profit Prediction System")
st.markdown("Enter your startup's spending details to predict profit.")

col1, col2 = st.columns(2)

with col1:
    rd = st.number_input("R&D Spend ($)", min_value=0.0,
                         value=75000.0, step=1000.0)
    admin = st.number_input("Administration ($)",
                            min_value=0.0, value=120000.0, step=1000.0)

with col2:
    marketing = st.number_input(
        "Marketing Spend ($)", min_value=0.0, value=200000.0, step=1000.0)
    state = st.selectbox("State", ["California", "Florida", "New York"])

if st.button("Predict Profit", use_container_width=True):
    try:
        d1 = {
            'R&D Spend': [rd],
            'Administration': [admin],
            'Marketing Spend': [marketing],
            'State': [state]
        }

        newdf = pd.DataFrame(d1)
        newdf[oh.get_feature_names_out()] = oh.transform(newdf[['State']])
        newdf.drop(columns=oh.feature_names_in_, inplace=True)
        newdf[['R&D Spend', 'Administration', 'Marketing Spend']] = sc.transform(
            newdf[['R&D Spend', 'Administration', 'Marketing Spend']]
        )

        result = reg.predict(newdf)
        profit = result[0]

        st.success(f"💰 Predicted Profit: **${profit:,.2f}**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
