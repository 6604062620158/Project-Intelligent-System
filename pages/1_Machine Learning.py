import streamlit as st

from PIL import Image

st.title("แนวทางการพัฒนาโมเดล Machine Learning")

st.write("""
ขั้นตอนแรกที่สำคัญในการพํฒนาโมเดล Machine Learning คือการเตรียมข้อมูล Dataset  โดย Dataset ที่เราสนใจนำมาพัฒนาตัวโมเดล Machine Learning คือ Dataset สำหรับการทำนายการเกิดหมอก  เพราะการพยากรณ์สภาพอากาศ เช่น หมอก มีผลกระทบต่อความปลอดภัยในการเดินทางในชีวิตประจำวันหรือการท่องเที่ยว
""")

st.write("""
แหล่งที่มาของ Dataset  นำมาจาก Microsoft Copilot ซึ่งช่วยสร้างและแนะนำข้อมูลที่สอดคล้องกับความต้องการ โดย Dataset นี้ประกอบด้วย  ข้อมูล  19  แถว และมี 3 ฟีเจอร์ คือ Temperature, Humidity, Fog และยังเป็น Dataset ที่ยังไม่สมบูรณ์
""")

st.write("""
อัลกอริทึมที่ใช้พัฒนาโมเดล Machine Learning  ในโปรเจคจะมี 2 อัลกอริทึม คือ Random Forest และ Gradient Boosting    โดยจะใช้  Library  scikit-learn และเลือกใช้อัลกอริทึมที่มีประสิทธิภาพที่สุด จากผลของการใช้ Dataset  ฝึกเทรนโมเดลด้วยอัลกอริทึม ทั้ง 2 
""")

st.header("ทฤษฤษฎีของอัลกอริทึม Random Forest")
st.write("""
Random forest เป็นหนึ่งในกลุ่มของโมเดลที่เรียกว่า Ensemble learning ที่มีหลักการคือการเทรนโมเดลที่เหมือนกันหลายๆ ครั้ง (หลาย Instance) บนข้อมูลชุดเดียวกัน โดยแต่ละครั้งของการเทรนจะเลือกส่วนของข้อมูลที่เทรนไม่เหมือนกัน แล้วเอาการตัดสินใจของโมเดลเหล่านั้นมาโหวตกันว่า Class ไหนถูกเลือกมากที่สุด
ฟังดูเหมือนไม่น่าจะดี แต่ในความเป็นจริงแล้ว กลไกการรวมการตัดสินใจของผู้ตัดสินใจจำนวนมากเข้าด้วยกันมักจะให้ผลการตัดสินใจที่แม่นยำมากกว่าการพึงพาการตัดสินใจจากแหล่งเดียว ปรากฏการณ์นี้เป็นจริงในหลายมิติ เช่นในทางสังคม เราเรียกว่า "ปัญญาของฝูงชน" (Wisdom of the crowd) ซึ่งหากสนใจสามารถค้นคำนี้อ่านได้ทั่วไป
การเรียนรู้แบบ Ensemble นี้จะทำงานได้ดีบนเงื่อนไขที่ว่า โมเดลผู้ทำนายแต่ละตัวจะต้องเรียนรู้อย่างเป็นอิสระต่อกันให้มากที่สุด เหมือนกับเงื่อนไขของปัญญาของฝูงชน ว่าคนแต่ละคนจะต้องตัดสินใจด้วยตนเองให้มากที่สุดโดยไม่ได้รับข้อมูลจากคนอื่นหรือนำเอาข้อมูลจากคนอื่นมาเป็นส่วนในการตัดสินใจ
ใน Machine learning algorithm เรามีวิธีการที่ทำให้การตัดสินใจของแต่ละโมเดลเป็นอิสระต่อกัน โดยการใช้ Algorithm เดียวกัน แต่ให้แต่ละ Instance เรียนรู้จากส่วนของข้อมูลที่ไม่เหมือนกันโดยใช้การสุ่มเลือก กลไกนี้เรียกว่า Bagging และ Pasting โดยสิ่งที่ต่างกันคือ Bagging สามารถสุ่มเลือกข้อมูลรายการเดียวกันได้ แต่ Pasting ไม่อนุญาตให้สุ่มรายการซ้ำกันได้เลย ในทางปฏิบัติ Bagging จะลด Variance ของโมเดลได้ดีกว่า เพราะมีการเลือกรายการข้อมูลซ้ำ ทำให้ได้โมเดลที่เสถียรกว่าและมักจะแม่นยำกว่า Pasting
สำหรับ scikit-learn โมเดลแบบ Bagging จะจะใช้วิธีการสุ่มเลือกรายการข้อมูลแบบ Bootstrap โดยแต่ละ Instance จะเลือก 63% ของข้อมูล เหลือ 37% ที่แต่ละ Instance ไม่เห็น เราเรียก 37% นี้ว่า Out-of-bag (oob) instance การที่มี oob instance ทำให้เราสามารถประเมินความแม่นยำเฉลี่ยของทุกๆ Instance ได้ในระหว่างเทรน โดยการเรียก Argument oob_score=True ใน Classifier instance

""")

st.header("ทฤษฤษฎีของอัลกอริทึม  Gradient Boosting ")
st.write("""
Boosting เป็นอีกเทคนิคใน Ensemble learning ที่ใช้ Classifier หลายๆ Instance มาช่วยกันสร้างโมเดลและพยากรณ์
การอธิบาย Boosting ให้เข้าใจง่าย น่าจะลองเปรียบเทียบว่ามันต่างกับ Random forest อย่างไร ทั้งคู่เป็น Ensemble learning เหมือนกัน โดย Random forest จะใช้ Classifier หลาย Instance สร้างโมเดลและทำนายพร้อมกัน โดยใช้ "กฎของจำนวนขนาดใหญ่" (Law of large numbers) เป็นคุณสมบัติที่ทำให้การทำนายนั้นแม่นยำกว่าการใช้ Classifier เดี่ยวๆ
ส่วน Boosting นำ Classifier หลายตัวมาทำงานเป็นโซ่ต่อกัน โดยแต่ละตัวจะแก้ไขจุดด้อยของ Classifier ตัวก่อนหน้า พอเทรนเสร็จแล้ว Classifier ทุกตัวจะพยากรณ์ร่วมกัน
ในบทนี้จะแนะนำ Boosting algorithm ที่เป็นที่นิยมสองตัว คือ AdaBoost และ GradientBoosting

Gradient boosting เลือกวิธีการในการ Optimise อีกวิธี โดยการพยายามให้ Classifier instance ที่มาใหม่แต่ละตัว มีความแม่นยำขึ้นเรื่อยๆ โดยเรียนรู้จากค่าความคลาดเคลื่อนสะสมที่เกิดจากการทำนายของ Instance ก่อนหน้า
การทำงานของ Gradient boosting เข้าใจไม่ยาก สมมุติว่าเราใช้ DecisionTreeRegressor เราสามารถจำลอง Algorithm นี้ได้จากโค้ดดังนี้:
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max-depth=2)
tree_reg1.fit(X, y)
จากนั้นเราเทรน DecisionTreeRegressor instance ที่ 2 จากความคลาดเคลื่อนสะสม ซึ่งก็คือความต่างระหว่าง y^ กับ y:
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max-depth=2)
tree_reg2.fit(X, y2)
และทำแบบนี้อีกครั้ง:
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max-depth=2)
tree_reg3.fit(X, y3)
ส่วนเวลาพยากรณ์ เราจะเอาคำตอบของการพยากรณ์ทุกๆ Instance มาบวกกัน ซึ่งก็คือ:
h(x)=h1(x)+h2(x)+⋯+hJ(x)        
โดย J คือจำนวน Instance ทั้งหมด
แต่ในความเป็นจริงเราไม่ต้องโค้ดเองแบบนี้ สามารถใช้ GradientBoostingRegressor หรือ GradientBoostingClassifier class ในโมดูล sklearn.ensemble ของ scikit-learn ได้เลย
ในแต่ละรอบ Classifier instance จะเทรนจาก Input X เดียวกัน แต่เปลี่ยน y ให้เป็นความต่างของค่าพยากรณ์กับคำตอบจริง ระหว่าง Instance ก่อนหน้ากับ Instance ปัจจุบัน (Residual error)

""")



st.header(" ขั้นตอนการพัฒนาโมเดล")

st.write("""
ในการพัฒนาโมเดล เราจะพัฒนาโมเดล ในเว็บไซต์ https://colab.research.google.com/  ซึ่ง Google Colab (หรือที่เรียกว่า Colaboratory) เป็นเครื่องมือออนไลน์ที่ให้ผู้ใช้งานสามารถเขียนและรันโค้ด Python ได้ในสภาพแวดล้อมแบบ Jupyter Notebook โดยไม่ต้องติดตั้งซอฟต์แวร์เพิ่มเติมบนคอมพิวเตอร์ ซึ่งเหมาะสำหรับงานต่างๆ เช่น การวิเคราะห์ข้อมูล การพัฒนา Machine Learning และการทดลองเขียนโค้ด
""")

st.write("""
                ขั้นนตอนแรก คือ การเตรียมข้อมูลที่จะใช้ฝึกและทดสอบอัลกอริทึม (Data Preparation) 
นำข้อมูลที่เรานำมาจาก Microsoft Copilot สร้างเป็นไฟล์ Dataset เช่น incomplete_weather_data.csv    ใช้ โค้ด files.upload() เพื่ออัปโหลดไฟล์ incomplete_weather_data.csv ไปยัง Colab โหลดข้อมูลเข้าเป็น DataFrame ด้วย pandas    เพื่อความแม่นยำในการทำนายของโมเดล  เราจึงเพิ่มปริมาณข้อมูลสำหรับการเรียนรู้  โดยการเพิ่ม ข้อมูลใหม่ในรูปแบบ DataFrame และ การสร้างข้อมูลสังเคราะห์ ข้อมูลสังเคราะห์ถูกสร้างขึ้นเพื่อเลียนแบบข้อมูลที่มีอยู่ใน Dataset โดยใช้วิธีสุ่มค่าในช่วงที่เหมาะสม มีการใช้เงื่อนไข (Rules) เพื่อกำหนดค่า Fog ให้สมจริง สุดท้าย การรวมข้อมูลเข้ากับ Dataset เดิม ข้อมูลที่สร้างขึ้น (ทั้งข้อมูลใหม่และข้อมูลสังเคราะห์) ถูกผนวกเข้ากับ Dataset เดิมโดยใช้ฟังก์ชัน pd.concat() การเพิ่มข้อมูลช่วยเพิ่มตัวอย่างใน Dataset ซึ่งมีประโยชน์ต่อการฝึกโมเดล Machine Learning โดยเฉพาะเมื่อ Dataset เดิมมีขนาดเล็ก
จากนั้น  เนื่องจากข้อมูล Dataset ที่นำมายังไม่สมบูรณ์ เราจึงสร้างรายงานข้อมูล ใช้ไลบรารี ydata-profiling เพื่อสร้างรายงานเกี่ยวกับข้อมูล เพื่อเช็คข้อมูล การหาค่าที่สูญหาย และสถิติของฟีเจอร์   เมื่อเราพบว่า มีค่าที่สูญหาย ใน Dataset
เราต้องจัดการข้อมูลที่สูญหาย ด้วยการนำเข้า (import) KNNImputer จาก โมดูล impute ของ ไลบรารี sklearn เพื่อเติมค่าที่สูญหายใน Dataset โดยใช้ค่าเฉลี่ยข้อมูลที่คล้ายคลึงที่สุด 3 ตัว ใน Dataset   เราจะทำความสะอาดข้อมูล ด้วยการ กรองค่าที่ไม่สมเหตุสมผล ลบแถวข้อมูลที่มีค่าไม่สมเหตุสมผล เช่น อุณหภูมิต่ำกว่า -30 หรือสูงกว่า 50 องศา

""")

st.write("""

            ขั้นตอนที่สอง คือ การแบ่งข้อมูล (Data Splitting) เพื่อใช้ในการ ฝึกและทดสอบโมเดล 
กำหนด Features และ Target Features คือ ฟีเจอร์ Temperature และ Humidity และ   Target คือ Fog (0: ไม่มีหมอก, 1: มีหมอก)  สุดท้ายคือ แบ่งข้อมูล Training และ Testing  ใช้ train_test_split() เพื่อแบ่งข้อมูลเป็น 70% สำหรับการ Train และ 30% สำหรับการ Test  

""")

st.write("""
            ขั้นตอนที่สาม คือ การสร้างและทดสอบโมเดล (Model Development and Evaluation)         
โมเดล อัลกอริทึม Random Forest ใช้ RandomForestClassifier จาก scikit-learn  เทรนโมเดลด้วยข้อมูล Training Set ทำนายผลด้วยข้อมูล Testing Set  คำนวณประสิทธิภาพของโมเดล โดยใช้ Accuracy, Confusion Matrix, Classification Report และ Cross-Validation Score
         
โมเดล อัลกอริทึม Gradient Boosting ใช้ GradientBoostingClassifier จาก scikit-learn และ ทำกระบวนการเดียวกันกับ Random Forest (เทรน, ทำนาย, ประเมินผล)
""")

st.write("""
ขั้นตอนที่สี่ คือ การเปรียบเทียบผลลัพธ์ (Model Comparison)
เปรียบเทียบค่า Accuracy ระหว่าง Random Forest และ Gradient Boosting  ใช้ Cross-Validation เพื่อประเมินประสิทธิภาพของโมเดลเพิ่มเติม วิเคราะห์ Confusion Matrix และ Classification Report เพื่อดูข้อดีข้อเสียของแต่ละโมเดล   

""")

st.header("ข้อมูลผลลัพธ์ (Output Information) ")
st.write("")
st.write("")

st.markdown("**ผลลัพธ์ของการ Training และ Testing ของอัลกอริทึม Random Forest**")

# โหลดภาพจากโฟลเดอร์
image_path = "images/RandomForest.png"
image = Image.open(image_path)

# แสดงภาพ
st.image(image, caption="ภาพข้อมูลผลลัพธ์", use_container_width=True)



st.write("""
นี่คือการวิเคราะห์ผลลัพธ์ของ Random Forest ที่ได้จาก Accuracy, Classification Report, Confusion Matrix, และ Cross-Validation Score:
1. Accuracy
 	ค่า Accuracy = 0.92 (92%) แสดงว่าโมเดลสามารถพยากรณ์ผลได้ถูกต้องถึง 92% ของตัวอย่างทั้งหมด (13 ตัวอย่าง)
 	ค่าความแม่นยำนี้บ่งบอกว่าโมเดลทำงานได้ดีในภาพรวม
2. Classification Report
 	Class 0 (ไม่มีหมอก):
 	Precision = 1.00: ทุกครั้งที่โมเดลพยากรณ์ว่าไม่มีหมอก (Class 0) ผลลัพธ์ถูกต้อง 100%
 	  Recall = 0.90: โมเดลสามารถจับข้อมูล Class 0 ได้ครบถึง 90% ของทั้งหมด (10 ตัวอย่าง)
 	F1-Score = 0.95: โมเดลมีสมดุลระหว่าง Precision และ Recall ใน Class 0
 	Class 1 (มีหมอก):
 	Precision = 0.75: ผลลัพธ์การพยากรณ์ว่าเป็นหมอกถูกต้อง 75%
 	Recall = 1.00: โมเดลสามารถจับข้อมูลที่เป็นหมอกได้ครบทุกตัวอย่าง (3 ตัวอย่าง)
 	F1-Score = 0.86: โมเดลทำงานได้ดีในคลาสนี้ แต่อาจมี False Positives ที่ส่งผลต่อ Precision
 	Macro Average:
 	ค่าเฉลี่ยของ Precision, Recall, และ F1-Score สำหรับทุกคลาส (ไม่คำนึงถึงจำนวนตัวอย่างในแต่ละคลาส)
 	Macro F1-Score = 0.90
 	Weighted Average:
 	ค่าเฉลี่ยที่คำนึงถึงจำนวนตัวอย่างในแต่ละคลาส
 	Weighted F1-Score = 0.93
3. Confusion Matrix
 	ตารางแสดงผลลัพธ์:
 	[[9 1]
 	 [0 3]]
 	True Positives (TP) = 9: โมเดลพยากรณ์ว่าไม่มีหมอกถูกต้อง (Class 0)
 	False Positives (FP) = 1: โมเดลพยากรณ์ว่าไม่มีหมอกผิด (Class 0) แต่จริงๆ มีหมอก
 	True Negatives (TN) = 3: โมเดลพยากรณ์ว่ามีหมอกถูกต้อง (Class 1)
 	False Negatives (FN) = 0: โมเดลพยากรณ์ว่ามีหมอกผิด (Class 1) แต่จริงๆ ไม่มีหมอก
 	สรุป: โมเดลมีข้อผิดพลาดเพียง 1 ตัวอย่าง (False Positive) ซึ่งแสดงว่าโมเดลทำงานได้ดีมาก
4. Cross-Validation Score
 	ค่าเฉลี่ยของ Cross-Validation Score = 0.93 (93%)
 	บ่งบอกว่าโมเดลมีความเสถียรและสามารถพยากรณ์ได้ดีในหลายชุดข้อมูล

""")

st.write("")
st.write("")
st.write("")


st.markdown("**ผลลัพธ์ของการ Training และ Testing ของอัลกอริทึม Gradient Boosting**")

# โหลดภาพจากโฟลเดอร์
image_path = "images/GradientBoosting.png"
image = Image.open(image_path)

# แสดงภาพ
st.image(image, caption="ภาพข้อมูลผลลัพธ์", use_container_width=True)

st.write("""
นี่คือการวิเคราะห์ผลลัพธ์ของ Gradient Boostingที่ได้จาก Accuracy, Classification Report, Confusion Matrix, และ Cross-Validation Score
1. Accuracy
 	ค่า Accuracy = 1.00 (100%) แสดงว่าโมเดลสามารถพยากรณ์ผลได้ถูกต้องในทุกตัวอย่างของข้อมูล (13 ตัวอย่าง)
 	เป็นค่า Accuracy ที่สมบูรณ์แบบ บ่งบอกว่าโมเดลทำงานได้อย่างยอดเยี่ยมในข้อมูลชุดนี้
2. Classification Report
 	ทั้งสองคลาส (Class 0: ไม่มีหมอก และ Class 1: มีหมอก) มีค่า Precision, Recall และ F1-Score = 1.00
 	Precision = 1.00: โมเดลทำนายได้ถูกต้องทุกครั้งเมื่อระบุว่าเป็นคลาสใดคลาสหนึ่ง
 	Recall = 1.00: โมเดลสามารถจับตัวอย่างในแต่ละคลาสได้ครบทุกตัวอย่าง
 	F1-Score = 1.00: โมเดลมีความสมดุลระหว่าง Precision และ Recall ในระดับสูงสุด
 	Macro Average และ Weighted Average:
 	ทั้ง Macro และ Weighted Average ของ Precision, Recall และ F1-Score เท่ากับ 1.00 ซึ่งแสดงถึงประสิทธิภาพที่ดีในทุกคลาส
3. Confusion Matrix
 	ตารางแสดงผลลัพธ์:
 	[[10  0]
 	 [ 0  3]]
 	True Positives (TP) = 10: โมเดลพยากรณ์ว่าไม่มีหมอก (Class 0) ถูกต้อง
 	True Negatives (TN) = 3: โมเดลพยากรณ์ว่ามีหมอก (Class 1) ถูกต้อง
 	False Positives (FP) = 0: ไม่มีกรณีที่โมเดลพยากรณ์ผิดว่าเป็น Class 0
 	False Negatives (FN) = 0: ไม่มีกรณีที่โมเดลพยากรณ์ผิดว่าเป็น Class 1
 	สรุป: โมเดล Gradient Boosting ไม่มีข้อผิดพลาดในชุดข้อมูลนี้ ซึ่งเป็นผลลัพธ์ที่ยอดเยี่ยม
4. Cross-Validation Score
 	ค่าเฉลี่ยของ Cross-Validation Score = 0.90 (90%)
 	แม้ว่าค่า Accuracy จะสมบูรณ์แบบในชุดข้อมูลที่ใช้ทดสอบ แต่ Cross-Validation Score ที่ 90% แสดงว่าโมเดลอาจมีผลลัพธ์ที่แปรผันเล็กน้อยในข้อมูลชุดอื่นๆ

""")

st.write("")
st.write("")
st.write("")


st.markdown("**ข้อมูลเปรียบเทียบ Classification Report ระหว่าง Random Forest และ Gradient Boosting**")

# โหลดภาพจากโฟลเดอร์
image_path = "images/R&G.png"
image = Image.open(image_path)

# แสดงภาพ
st.image(image, caption="ภาพข้อมูลผลลัพธ์", use_container_width=True)

st.write("")
st.write("""
นี่คือการวิเคราะห์เปรียบเทียบ Classification Report ระหว่าง Random Forest และ Gradient Boosting:
1. Precision
	Random Forest:
	Class 0: 1.00 (ไม่มี False Positive สำหรับ Class 0)
	Class 1: 0.75 (มี False Positive สำหรับ Class 1)
	Precision เฉลี่ย (Weighted Avg): 0.94
	Gradient Boosting:
	ทั้ง Class 0 และ Class 1: 1.00 (ไม่มี False Positive ในทั้งสองคลาส)
	Precision เฉลี่ย (Weighted Avg): 1.00
	สรุป: Gradient Boosting ทำงานได้ดีกว่าในแง่ของ Precision สำหรับ Class 1 เนื่องจากไม่มีข้อผิดพลาด False Positive
2. Recall
	Random Forest:
	Class 0: 0.90 (จับ Class 0 ได้ 90% ของทั้งหมด)
	Class 1: 1.00 (จับ Class 1 ได้ครบทุกตัวอย่าง)
	Recall เฉลี่ย (Weighted Avg): 0.92
	Gradient Boosting:
	ทั้ง Class 0 และ Class 1: 1.00 (จับทุกตัวอย่างในทุกคลาสได้ถูกต้อง)
	Recall เฉลี่ย (Weighted Avg): 1.00
	สรุป: Gradient Boosting เหนือกว่า Random Forest เพราะจับทุกตัวอย่างในแต่ละคลาสได้ครบ
""")

st.header("สรุปผลข้อดีข้อเสียของอัลกอริทึม Random Forest และ Gradient Boosting และเลือกใช้อัลกอริทึมที่เหมาะสม")


st.write("""
ข้อดีและข้อเสียที่สังเกตได้
1.	Random Forest:
	ข้อดี:
	เหมาะกับการจัดการข้อมูลที่ไม่สมดุล
	ต้านทาน Overfitting ได้ดี
	ข้อเสีย:
	ประสิทธิภาพอาจไม่สูงเท่า Gradient Boosting ในกรณีข้อมูลซับซ้อน
2.	Gradient Boosting:
	ข้อดี:
	ทำงานได้ดีมากกับข้อมูลซับซ้อน และมีประสิทธิภาพสูงในการจับความสัมพันธ์ในข้อมูล
	ข้อเสีย:
	ใช้เวลาและทรัพยากรมากกว่า อาจเกิด Overfitting หากปรับพารามิเตอร์ไม่ดี

""")

st.write("""
 สรุปผล จากการพัมนาโมเดล Machine-Learning ด้วย 2 โมเดลและอัลกอริทึม  Random Forest กับ Gradient Boosting
 จากข้อมูลและคะแนนการฝึก แสดงให้เห็นว่าโมเดล Gradient Boosting มีประสิทธิภาพสูงสุดในการทำนายการเกิดหมอก
จึงเลือกใช้ Gradient Boosting ในการพยากรณ์การเกิดหมอกในอนาคต ของโมเดล Machine-Learning ที่จะพัฒนาต่อไป

""")


st.header("แหล่งอ้างอิงข้อมูล")

st.write(""" 
 แหล่งข้อมูลอ้างอิง ทฤษฤษฎีของอัลกอริทึม Random Forest
 โดย ชิตพงษ์ กิตตินราดร | มกราคม 2563
แหล่งที่มา    https://shorturl.asia/7wvs4
         
แหล่งข้อมูลอ้างอิง ทฤษฤษฎีของอัลกอริทึม Gradient Boosting
โดย ชิตพงษ์ กิตตินราดร | มกราคม 2563
แหล่งที่มา  https://guopai.github.io/ml-blog11.html


 แหล่งข้อมูลอ้างอิง ข้อมูลสร้างโปรเจค และ Dataset   
    โดย  Microsoft Copilot 


""")