# รายงานฉบับร่าง

ประกอบการสอบ โครงงานวิศวกรรมคอมพิวเตอร์ 2 ปีการศึกษา 2568
สาขาวิชาวิศวกรรมคอมพิวเตอร์ คณะเทคโนโลยีดิจิทัล สถาบันเทคโนโลยีจิตรลดา

## ระบบเปรียบเทียบลายนิ้วมือด้วย Deep Learning (Fingerprint Verification with Siamese Neural Network) — V4

**ชื่อ-นามสกุล นักศึกษา:** [นาย ณกรณ์ ปภาสิทธิมงคล]   **อาจารย์ที่ปรึกษา:** [อาจารย์ กฤษฎา พรหมสุทธิรักษ์ ]

---

## 1. บทนำ (Introduction)

### 1.1 ที่มาและความสำคัญ
ลายนิ้วมือเป็นข้อมูลที่ใช้ยืนยันตัวตนของคนได้แม่นยำ เพราะลายนิ้วมือของแต่ละคน "ไม่ซ้ำกัน" ปัจจุบันจึงนิยมใช้ในมือถือ ATM และระบบเข้า-ออกอาคาร แต่ระบบเดิมที่ใช้วิธี **จับจุดสันลายนิ้วมือ (minutiae matching)** มักทำงานไม่ดีเมื่อภาพไม่ชัด มีรอยขีดข่วน หรือนิ้วเปียก ปัจจุบันเทคโนโลยี **Deep Learning** โดยเฉพาะแบบ **Siamese Neural Network** สามารถเรียนรู้ลักษณะของลายนิ้วมือเองได้ และเปรียบเทียบภาพ 2 ภาพได้โดยตรง

ปัญหาที่พบในงานวิจัยจำนวนมากคือทดสอบโมเดลกับ "คนเดิม" ที่ใช้ฝึก ทำให้ผลลัพธ์ดูดีเกินจริง โครงงานนี้จึงเน้นการทดสอบแบบ **subject-disjoint** คือทดสอบกับคนที่โมเดลไม่เคยเห็นมาก่อน เพื่อให้ผลใกล้เคียงการใช้งานจริง

### 1.2 วัตถุประสงค์ (SMART Goals)
1. พัฒนาโมเดล Siamese ที่ตอบคะแนนความเหมือน (similarity score) ของคู่ลายนิ้วมือใด ๆ ได้ **AUC ≥ 0.95** บน test set ที่เป็นคนใหม่ทั้งหมด
2. ทำให้ **False Accept Rate (FAR) ≤ 1%** เพื่อความปลอดภัย
3. รองรับกรณียาก (ภาพถูกแก้ไข / นิ้วเดียวกันแต่ภาพต่างกัน) ให้มี **FRR ≤ 20%**
4. พัฒนา REST API ที่ตอบกลับภายใน **500 ms ต่อ request**
5. ประเมินผลตามมาตรฐาน **ISO/IEC 19795** (DET curve, EER, FNMR@FMR)

### 1.3 ขอบเขตของโครงงาน
**Features หลัก:**
- รับภาพลายนิ้วมือ 2 ภาพ ตอบ similarity score (0–1)
- API endpoints: `/verify`, `/enroll`, `/stats`
- สคริปต์ปรับ threshold (จุดตัด) ให้เหมาะกับแต่ละ use case
- ระบบประเมินผลครบ (ROC, DET, PR, Confusion Matrix, Threshold sweep)

**Constraints & Assumptions:**
- ใช้ dataset SOCOFing-style (Real + Altered-Hard) ภาพ grayscale 90×90 px
- รันบน Windows 11 + Python 3.10 + TensorFlow 2.x
- ไม่รองรับ multi-finger fusion และไม่ออกแบบ hardware sensor
- ทดสอบเฉพาะกรณี 1-to-1 verification (ไม่ใช่ 1-to-many identification)

### 1.4 ประโยชน์ที่คาดว่าจะได้รับ
- **เชิงปริมาณ:** ลดต้นทุนเทียบกับ commercial SDK (open-source, ฟรี), inference ~200 ms, FAR ต่ำเพียง 0.10%
- **เชิงคุณภาพ:** เป็น baseline open-source สำหรับการศึกษา/วิจัยต่อยอด, ใช้เป็นกรณีศึกษาเรื่อง **data leakage** และ **open-set evaluation** ในงาน biometric

---

## 2. ทฤษฎีและเทคโนโลยีที่เกี่ยวข้อง

### 2.1 ทฤษฎีที่เกี่ยวข้อง
- **Siamese Neural Network:** เครือข่ายประสาทเทียมแบบ 2 แขนที่ **แชร์ weights กัน** รับ input 2 ภาพ คำนวณ feature ของแต่ละภาพแล้วเปรียบเทียบ — ใช้เป็นแกนหลักของโมเดล
- **Convolutional Neural Network (CNN):** ใช้สกัด feature จากภาพ (V4 ใช้ 3 บล็อก Conv+MaxPool)
- **Binary Cross-Entropy Loss:** ฟังก์ชัน loss สำหรับงาน binary (ใช่/ไม่ใช่)
- **Data Augmentation:** เพิ่มความหลากหลายของข้อมูล (rotate, brightness) เพื่อลด overfitting
- **Subject-disjoint / Open-set Evaluation:** แยกคนระหว่าง train และ test อย่างสมบูรณ์ — เป็นมาตรฐานสากลของงาน biometric

### 2.2 เปรียบเทียบเทคโนโลยีและเหตุผลการเลือก

| หัวข้อ | ทางเลือก | เลือก & เหตุผล |
|---|---|---|
| สถาปัตยกรรมโมเดล | Siamese / Triplet / ArcFace | **Siamese** — implement ง่าย, ใช้ pair-label ตรง |
| Framework | TensorFlow/Keras / PyTorch | **TensorFlow 2.x** — deploy `.h5` ง่าย, Keras API กระชับ |
| Loss Function | BCE / Contrastive / Triplet | **BCE** — ตรงกับ sigmoid output, debug ง่าย |
| Web Framework | FastAPI / Flask / Django | **FastAPI** — เร็ว + Swagger auto docs |

---

## 3. ระเบียบวิธีวิจัยและการออกแบบระบบ

### 3.1 สถาปัตยกรรมระบบ (System Architecture)
```
[Client]  ──HTTP POST──►  [FastAPI Server]  ──►  [Siamese Model V4 (.h5)]
                                │                          │
                                ▼                          ▼
                          [Template DB]              [Score 0–1]
                                                           │
                                                           ▼
                                              [เทียบ Threshold → ใช่/ไม่ใช่]
```
**Data flow:** Client ส่งภาพ 2 ภาพ → API preprocess (resize 90×90, normalize) → โมเดลคำนวณ score → เทียบกับ threshold → ส่งผลกลับเป็น JSON

### 3.2 การออกแบบรายละเอียด
**โมเดล V4:**
- Input: 2 × (90×90×1) grayscale
- Shared CNN: 3 บล็อก (Conv 3×3 → ReLU → MaxPool)
- Lambda Layer: `|feat₁ − feat₂|` (absolute difference)
- Head: Conv → Flatten → Dense(64) → Dropout(0.3) → Dense(1, sigmoid)

**API Endpoints:**
- `POST /verify` — รับ 2 ภาพ คืน score + decision
- `POST /enroll` — ลงทะเบียน template
- `GET /stats` — สถิติการใช้งาน

### 3.4 กระบวนการพัฒนา
ใช้ **iterative development** แบ่งเป็น V1 → V2 → V3 → V4 — แต่ละเวอร์ชันแก้ปัญหาของเวอร์ชันก่อน เช่น **V3** แก้ปัญหา data leakage ของ V2 และ **V4** เพิ่มสัดส่วน hard-negative เป็น 50% เพื่อแก้ปัญหา "นิ้วเดียวกันคนเดียวกัน" ที่ V3 ตอบผิด ใช้เครื่องมือ: **Git** (version control), **Jupyter Notebook** (experiment), **VS Code** (coding)

---

## 4. การดำเนินการ / การทดลอง

### 4.1 ระเบียบวิธีทดสอบ
- **สภาพแวดล้อม:** Windows 11, Python 3.10, TensorFlow 2.x
- **Subject-disjoint split:** 600 คน (train) / 200 คน (test, held-out)
- **Test pairs:** 4,000 คู่ (genuine 50% + impostor 50%)
- **Hard test:** same-person-different-finger 2,000 คู่ (เคสที่ V3 ทำพลาดบ่อย)
- **Performance test:** วัด response time ของ API
- **Unit / Integration test:** ครอบคลุม preprocessing, inference, API endpoints

### 4.2 ตัวชี้วัดความสำเร็จ
ใช้ตัวชี้วัดตามมาตรฐาน **ML** และ **biometric ISO/IEC 19795**:
- **AUC** (Area Under ROC Curve) — ความสามารถแยก genuine/impostor
- **EER** (Equal Error Rate) — จุดที่ FAR = FRR (ตัวเลขหลักของงาน biometric)
- **FAR / FRR / Precision / Recall / F1**
- **Confusion Matrix, ROC, DET, PR Curve**
- **FNMR @ Fixed FMR** (operating point มาตรฐานอุตสาหกรรม)
- **d-prime** — ระยะห่างของ score distribution

---

## 5. ผลการทดลองและการวิเคราะห์

### 5.1 ผลการทดสอบ

| ตัวชี้วัด | เป้าหมาย | ผล V4 | ผ่าน |
|---|---|---|---|
| AUC | ≥ 0.95 | **0.9927** | ✅ |
| EER | — | 4.13% | — |
| FAR @ threshold 0.70 | ≤ 1% | **0.10%** | ✅ |
| FRR @ threshold 0.70 | ≤ 20% | 18.13% | ✅ |
| Same-finger FAR (hard) | ≤ 1% | **0.50%** | ✅ |
| d-prime | — | 3.27 (ดี) | — |
| API response time | < 500 ms | ~200 ms | ✅ |

**กราฟที่ได้ครบ:** ROC curve (AUC 0.9927), DET curve, PR curve (AP 0.9937), Score distribution, Confusion Matrix, Threshold comparison

### 5.2 อภิปรายผล
**จุดแข็ง:**
- AUC = 0.9927 บน test set ที่เป็นคนใหม่ → โมเดล generalize ได้ดี
- FAR ต่ำมาก (0.10%) → เหมาะกับงาน access control
- แก้ปัญหา same-person-different-finger ของ V3 ได้ (FAR ลดจาก ~99% เหลือ 0.50%)

**จุดอ่อน:**
- FRR สูง (18.13%) ที่ threshold 0.70 → ผู้ใช้ต้องสแกนหลายครั้ง
- Score distribution เป็น **bimodal สุดขั้ว** (ส่วนใหญ่ตอบ 0 หรือ 1) → โมเดล overconfident
- จากการวิเคราะห์เพิ่ม พบว่า **threshold 0.70 ไม่ใช่จุดที่ดีที่สุด** — Youden's J ที่ ~0.00 ให้ Accuracy 96.15% และ F1 0.9614

### 5.3 ข้อจำกัดและความเสี่ยง
**ข้อจำกัด:**
- ทดสอบเฉพาะ dataset SOCOFing-style → ยังไม่ทราบประสิทธิภาพกับ sensor จริง
- ไม่รองรับ multi-finger fusion
- ทำ FAR < 0.01% ไม่ได้ → ไม่เหมาะกับงานความปลอดภัยสูงสุด เช่น ตม.

**ความเสี่ยงและแนวทางลด:**
- **Overconfidence:** โมเดลตอบ 0/1 สุดขั้ว → แก้ด้วย temperature scaling หรือ label smoothing ใน V5
- **Catastrophic errors:** มีคู่ genuine score = 0 และ impostor score ≈ 1 → วิเคราะห์ภาพ + retrain ด้วย hard examples
- **Generalization:** ทดสอบเฉพาะ dataset เดียว → ทดสอบเพิ่มกับ FVC, NIST SD ในอนาคต
