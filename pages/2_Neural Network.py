import streamlit as st

st.title("แนวทางการพํฒนาโมเดล Neural Network")

st.write("""
ขั้นตอนแรกที่สำคัญในการพํฒนาโมเดล Neural Networkคือการเตรียมข้อมูล Dataset  โดย Dataset ที่เราสนใจนำมาพัฒนาตัวโมเดล Neural Network คือ Dataset สำหรับการการจำแนกประเภท ชุดข้อมูลป้ายจราจร 
แหล่งที่มาของ Dataset  นำมาจาก  เว็บไซต์ https://www.kaggle.com/  โดย Dataset นี้ประกอบด้วย  ภาพ ชุดข้อมูลป้ายจราจร  57 ประเภท และมี 2 ฟีเจอร์ คือ ClassId ,Name  และยังเป็น Dataset ที่ยังไม่สมบูรณ์
อัลกอริทึมที่ใช้พัฒนาโมเดล Neural Network ในโปรเจค คือ Convolutional Neural Network (CNN)

""")




st.header("ทฤษฤษฎีของอัลกอริทึม Convolutional Neural Network (CNN)")

st.write("""
Convolutional Neural Network คืออะไร
Convolutional Neural Network (CNN) หรือ โครงข่ายประสาทแบบคอนโวลูชัน เป็นโครงข่ายประสาทเทียมหนึ่งในกลุ่ม bio-inspired โดยที่ CNN จะจำลองการมองเห็นของมนุษย์ที่มองพื้นที่เป็นที่ย่อยๆ และนำกลุ่มของพื้นที่ย่อยๆมาผสานกัน เพื่อดูว่าสิ่งที่เห็นอยู่เป็นอะไรกันแน่
การมองพื้นที่ย่อยของมนุษย์จะมีการแยกคุณลักษณะ (feature) ของพื้นที่
ย่อยนั้น เช่น ลายเส้น และการตัดกันของสี ซึ่งการที่มนุษย์รู้ว่าพื้นที่ตรงนี้เป็นเส้นตรงหรือสีตัดกัน เพราะมนุษย์ดูทั้งจุดที่สนใจและบริเวณรอบ ๆ ประกอบกัน
Feature Extraction
แนวคิดของ CNN นั้นค่อนข้างเป็นแนวคิดที่ดีมาก แต่สิ่งที่ซับซ้อนของมันคือระบบการคำนวณที่สอดคล้องกับ Concept ของมันเองและต้องมีคณิตศาสตร์มารองรับ โดยการคำนวณตามแนวคิดนี้ใช้หลักการเดียวกันกับ คอนโวลูชันเชิงพื้นที่ (Spatial Convolution) ในการทำงานด้าน Image Processing
การคำนวณนี้จะเริ่มจากการกำหนดค่าใน ตัวกรอง (filter) หรือ เคอร์เนล
(kernel) ที่ช่วยดึงคุณลักษณะที่ใช้ในการรู้จำวัตถุออก โดยปกติตัวกรอง/เคอร์เนลอันหนึ่งจะดึงคุณลักษณะที่สนใจออกมาได้หนึ่งอย่าง เราจึงจำเป็นต้องตัวกรองหลายตัวกรองด้วย เพื่อหาคุณลักษณะทางพื้นที่หลายอย่างประกอบกัน


""")

st.header(" ขั้นตอนการพัฒนาโมเดล")
st.write("""
ในการพัฒนาโมเดล เราจะพัฒนาโมเดล ในเว็บไซต์ https://colab.research.google.com/  ซึ่ง Google Colab (หรือที่เรียกว่า Colaboratory) เป็นเครื่องมือออนไลน์ที่ให้ผู้ใช้งานสามารถเขียนและรันโค้ด Python ได้ในสภาพแวดล้อมแบบ Jupyter Notebook โดยไม่ต้องติดตั้งซอฟต์แวร์เพิ่มเติมบนคอมพิวเตอร์ ซึ่งเหมาะสำหรับงานต่างๆ เช่น การวิเคราะห์ข้อมูล การพัฒนา Machine Learning และการทดลองเขียนโค้ด"
""")


st.write("""
ขั้นตอนแรก การตรวจสอบความสมบูรณ์ของข้อมูล (Data Integrity Check)
ข้อมูลใน Dataset จะแบ่งเป็น 3 ส่วน  1. DATA คือรูปภาพป้ายจราจร ส่วนที่ใช้ Training       2. TEST คือรูปภาพป้ายจราจร ส่วนที่ใช้  Testing     3.LABELS ส่วนที่เป็นส่วนเฉลยว่า  ภาพแต่ละภาพคือประเภทอะไร
เนื่องจากข้อมูล Dataset ที่นำมายังไม่สมบูรณ์ เริ่มจากการตรวจสอบข้อมูล ความไม่สมบูรณ์ ของ Dataset เพื่อเลือกวิธีจัดการทำความสะอาด  จากการตรวจสอบพบว่าข้อมูลขาดหายกลายเป็นข้อมูลที่ไม่รู้จัก เนื่องจากข้อมูลที่หายไปถูกแทน ด้วยUnknown 
เราจะทำความสะอาด  ด้วยการใส่ค่าข้อมูลที่ถูกต้อง โดยอิงจาก ฟีเจอร์ อื่นๆ

""")

st.write("""ขั้นตอนที่สอง คือ การเตรียมข้อมูล (Data Preprocessing):
         •  เป้าหมาย: แปลงรูปภาพและ Label ให้อยู่ในรูปแบบที่เหมาะสมสำหรับการฝึกโมเดล CNN.
•  กระบวนการ:
1.	สร้าง DataFrame ที่เชื่อมโยงชื่อไฟล์ (image_path) กับ ClassId (label).
2.	แบ่งข้อมูลออกเป็นชุด Training และ Validation ด้วย train_test_split.
3.	ใช้ฟังก์ชัน preprocess_images:
o	โหลดรูปภาพจากเส้นทางใน DataFrame.
o	ปรับขนาดรูปภาพให้อยู่ในขนาดคงที่ (64x64 พิกเซล).
o	Normalization ค่า pixel ให้อยู่ในช่วง 0 ถึง 1.
4.	แปลงค่าของ Label (ClassId) เป็น One-hot Encoding เพื่อให้เหมาะสมกับการคำนวณ categori-cal_crossentropy.
""")



st.write("""
ขั้นตอนที่สาม คือ การสร้างโมเดล CNN (Building the CNN Model):

•  เป้าหมาย: สร้างโมเดล Convolutional Neural Network (CNN) ที่เหมาะสำหรับการจำแนกประเภทภาพ.
•  โครงสร้างโมเดล:
1.	Convolutional Layers:
o	ดึงฟีเจอร์จากภาพโดยใช้เลเยอร์ Conv2D พร้อมขนาดฟิลเตอร์ที่กำหนด.
o	ลดขนาดภาพด้วย MaxPooling เพื่อเน้นส่วนสำคัญของภาพ.
2.	Flatten Layer:
o	แปลงข้อมูลที่เป็น 2D array ให้อยู่ในรูปแบบ 1D array เพื่อส่งเข้า Fully Connected Layer.
3.	Fully Connected Layers:
o	ใช้เลเยอร์ Dense เพื่อจับคู่ฟีเจอร์ที่ดึงออกมากับ ClassId.
o	ใช้ Dropout เพื่อลด Overfitting.
4.	Output Layer:
o	มี 58 โหนด (จำนวนหมวดหมู่ ClassId ทั้งหมด) พร้อม Activation แบบ softmax เพื่อให้ผลลัพธ์เป็นความน่าจะเป็นแต่ละ Class.
•  คอมไพล์โมเดล:
•	ใช้ Optimizer Adam สำหรับการปรับปรุงค่าพารามิเตอร์.
•	Loss Function: categorical_crossentropy สำหรับงานจำแนกประเภทแบบหลายหมวดหมู่.



""")




st.write("""
ขั้นตอนที่สี่ การฝึกโมเดล (Training the Model):
เป้าหมาย: ใช้ข้อมูล Training และ Validation เพื่อฝึกโมเดล.
กระบวนการ:ใช้ฟังก์ชัน fit เพื่อฝึกโมเดล.
ระบุจำนวน Epochs (รอบของการเรียนรู้) และ Batch Size.
ติดตามค่าความสูญเสีย (Loss) และความแม่นยำ (Accuracy) ระหว่างการฝึก.
""")




st.write("""
ขั้นตอนที่ห้า การเตรียมข้อมูลทดสอบ (Test Data Preparation):
เป้าหมาย: เตรียมชุดข้อมูลสำหรับการประเมินผลโมเดล.
กระบวนการ:สร้าง DataFrame สำหรับไฟล์ในโฟลเดอร์ TEST.
ใช้ฟังก์ชัน preprocess_images โหลดและปรับขนาดรูปภาพ.
แปลง Label ของข้อมูลทดสอบเป็น One-hot Encoding.
""")

st.write("""   
ขั้นตอนที่หก การประเมินผลโมเดล (Evaluating the Model):
   เป้าหมาย: ตรวจสอบความแม่นยำของโมเดลด้วยชุดข้อมูล TEST.
กระบวนการ:ใช้ฟังก์ชัน evaluate เพื่อคำนวณค่า loss และ accuracy จากข้อมูล x_test และ y_test. 
""")

st.write("""   
ขั้นตอนที่เจ็ด การแสดงผลลัพธ์และการเชื่อมโยง labels.csv:
เป้าหมาย: แปลง ClassId ที่โมเดลทำนายให้อยู่ในรูปแบบที่มนุษย์เข้าใจ.
กระบวนการ:ดึงค่าที่โมเดลคาดการณ์ (Predicted Classes).
ใช้ labels.csv เพื่อแปลง ClassId เป็นชื่อหมวดหมู่ของป้ายจราจร.
แสดงตัวอย่างการทำนาย 5 รายการ.

""")

st.write("""   
ขั้นตอนสุดท้าย การบันทึกโมเดลที่ฝึกเสร็จ (Saving the Model):
เป้าหมาย: เก็บโมเดลไว้สำหรับการใช้งานในอนาคต.
กระบวนการ:ใช้ฟังก์ชัน save เพื่อบันทึกโมเดลในรูปแบบไฟล์ .h5.

""")

st.header("แหล่งอ้างอิงข้อมูล")

st.write(""" 
 แหล่งข้อมูลอ้างอิง ทฤษฤษฎีของอัลกอริทึม Convolutional Neural Network (CNN)
 โดย Natthawat Phongchit
แหล่งที่มา     https://shorturl.asia/WHfuK
         
แหล่งข้อมูลอ้างอิง  Dataset สำหรับการการจำแนกประเภท ชุดข้อมูลป้ายจราจร 

แหล่งที่มา  https://www.kaggle.com/  


 แหล่งข้อมูลอ้างอิง ข้อมูลสร้างโปรเจค 
    โดย  Microsoft Copilot 


""")