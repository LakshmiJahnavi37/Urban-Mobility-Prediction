import os, sys
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_traffic

st.set_page_config(page_title="Traffic Flow Prediction", page_icon="🚦")
st.title("🚦 Traffic Flow Prediction")
st.caption("Predict hourly vehicle count at a junction")

c1, c2 = st.columns(2)
junction = c1.selectbox("Junction", [1, 2, 3, 4])
hour = c2.slider("Hour of day", 0, 23, 9)
dow = c1.selectbox("Day of week", range(7),
                   format_func=lambda i: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i])
month = c2.slider("Month", 1, 12, 6)
day = c1.slider("Day of month", 1, 31, 15)
st.subheader("Recent traffic (vehicles)")
lag1 = st.number_input("1 hour ago", value=40)
lag24 = st.number_input("Same hour yesterday", value=40)
lag168 = st.number_input("Same hour last week", value=40)

if st.button("Predict"):
    try:
        out = predict_traffic(junction, hour, dow, month, day, lag1, lag24, lag168)
        st.success(f"Predicted vehicles: {out:.0f}")
    except FileNotFoundError as e:
        st.error(str(e))
