import streamlit as st
import joblib
import pandas as pd

st.title("Demo การพยากรณ์หมอกด้วย Machine Learning")

st.write("""
ในการารพยากรณ์ สามารถทดลองใส่ค่าพารามิเตอร์ เช่น อุณหภูมิ (Temperature) และความชื้น (Humidity) 
เพื่อดูผลลัพธ์การพยากรณ์โอกาสการเกิดหมอกจากโมเดล Machine Learning ของเรา
""")


# โหลดโมเดล Gradient Boosting ที่บันทึกไว้
model = joblib.load('gradient_boosting_model.pkl')

# สร้างฟังก์ชันการทำนาย
def predict_fog(temperature, humidity):
    input_data = {'Temperature': [temperature], 'Humidity': [humidity]}
    input_df = pd.DataFrame(input_data)
    prediction = model.predict(input_df)
    return prediction[0]

# สร้างส่วนอินเตอร์เฟซผู้ใช้ด้วย Streamlit
st.title('Fog Prediction App')
st.write('กรุณาใส่อุณหภูมิและความชื้นเพื่อทำนายการเกิดหมอก')

# รับค่าจากผู้ใช้
temperature = st.number_input('Temperature (°C)', min_value=-50.0, max_value=50.0, value=20.0)
humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=85.0)

if st.button('Predict'):
    result = predict_fog(temperature, humidity)
    if result == 1:
        st.success('มีโอกาสเกิดหมอก')
    else:
        st.success('มีโอกาสเกิดหมอกต่ำหรือไม่มีโอกาสเกิดหมอกเลย')