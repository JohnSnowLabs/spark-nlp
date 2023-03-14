---
layout: model
title: Pdf processing
author: John Snow Labs
name: pdf_processing
date: 2023-01-03
tags: [en, licensed, ocr, pdf_processing]
task: Document Pdf Processing
language: en
nav_key: models
edition: Visual NLP 4.0.0
spark_version: 3.2.1
supported: true
annotator: PdfProcessing
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model obtain the text of an input PDF document with a Tesseract pretrained model. Tesseract is an Optical Character Recognition (OCR) engine developed by Google. It is an open-source tool that can be used to recognize text in images and convert it into machine-readable text. The engine is based on a neural network architecture and uses machine learning algorithms to improve its accuracy over time.

Tesseract has been trained on a variety of datasets to improve its recognition capabilities. These datasets include images of text in various languages and scripts, as well as images with different font styles, sizes, and orientations. The training process involves feeding the engine with a large number of images and their corresponding text, allowing the engine to learn the patterns and characteristics of different text styles. One of the most important datasets used in training Tesseract is the UNLV dataset, which contains over 400,000 images of text in different languages, scripts, and font styles. This dataset is widely used in the OCR community and has been instrumental in improving the accuracy of Tesseract. Other datasets that have been used in training Tesseract include the ICDAR dataset, the IIIT-HWS dataset, and the RRC-GV-WS dataset.

In addition to these datasets, Tesseract also uses a technique called adaptive training, where the engine can continuously improve its recognition capabilities by learning from new images and text. This allows Tesseract to adapt to new text styles and languages, and improve its overall accuracy.


## Predicted Entities

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-ocr-workshop/blob/master/tutorials/Certification_Trainings/2.1.Pdf_processing.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
pdf_to_text = PdfToText() \
    .setInputCol("content") \
    .setOutputCol("text") \
    .setSplitPage(True) \
    .setExtractCoordinates(True) \
    .setStoreSplittedPdf(True)

pdf_to_image = PdfToImage() \
    .setInputCol("content") \
    .setOutputCol("image") \
    .setKeepInput(True)

ocr = ImageToText() \
    .setInputCol("image") \
    .setOutputCol("text") \
    .setConfidenceThreshold(60)

pipeline = PipelineModel(stages=[
    pdf_to_text,
    pdf_to_image,
    ocr
])

pdf_path = pkg_resources.resource_filename('sparkocr', 'resources/ocr/pdfs/*.pdf')
pdf_example_df = spark.read.format("binaryFile").load(pdf_path).cache()

result = pipeline().transform(pdf_example_df).cache()
```
```scala
val pdf_to_text = new PdfToText() 
    .setInputCol("content") 
    .setOutputCol("text") 
    .setSplitPage(True) 
    .setExtractCoordinates(True) 
    .setStoreSplittedPdf(True)

val pdf_to_image = new PdfToImage() 
    .setInputCol("content") 
    .setOutputCol("image") 
    .setKeepInput(True)

val ocr = new ImageToText() 
    .setInputCol("image") 
    .setOutputCol("text") 
    .setConfidenceThreshold(60)

val pipeline = new PipelineModel().setStages(Array(
    pdf_to_text, 
    pdf_to_image, 
    ocr))

val pdf_path = pkg_resources.resource_filename("sparkocr", "resources/ocr/pdfs/*.pdf")
val pdf_example_df = spark.read.format("binaryFile").load(pdf_path).cache()

val result = pipeline().transform(pdf_example_df).cache()
```
</div>

## Example

### Input:
Representation of images 2 and 3 of the example:
![Screenshot](/assets/images/examples_ocr/image7.png)

## Output text

```bash
+--------------------+-------------------+------+--------------------+--------------------+-----------------+------------------+--------------------+--------------------+-----------+-------+-----------+----------------+---------+
|                path|   modificationTime|length|                text|           positions| height_dimension|   width_dimension|             content|               image|total_pages|pagenum|documentnum|      confidence|exception|
+--------------------+-------------------+------+--------------------+--------------------+-----------------+------------------+--------------------+--------------------+-----------+-------+-----------+----------------+---------+
|file:/Users/nmeln...|2022-07-14 15:38:51|693743|Patient Nam\nFina...|[{[{Patient Nam\n...|1587.780029296875|1205.8299560546875|[25 50 44 46 2D 3...|{file:/Users/nmel...|          1|      0|          0|81.2276874118381|     null|
|file:/Users/nmeln...|2022-07-14 15:38:51|693743|Patient Name\nFin...|[{[{Patient Name\...|1583.780029296875|1217.8299560546875|[25 50 44 46 2D 3...|{file:/Users/nmel...|          1|      0|          0|78.5234429732613|     null|
|file:/Users/nmeln...|2022-07-14 15:38:51| 70556|Alexandria is the...|[{[{A, 0, 72.024,...|            792.0|             612.0|[25 50 44 46 2D 3...|                null|       null|      0|          0|            null|     null|
|file:/Users/nmeln...|2022-07-14 15:38:51| 70556|Alexandria was fo...|[{[{A, 1, 72.024,...|            792.0|             612.0|[25 50 44 46 2D 3...|                null|       null|      0|          0|            null|     null|
|file:/Users/nmeln...|2022-07-14 15:38:51| 11601|8 i , . ! \n9 i ,...|[{[{8, 0, 72.0604...|            843.0|             596.0|[25 50 44 46 2D 3...|                null|       null|      0|          0|            null|     null|
+--------------------+-------------------+------+--------------------+--------------------+-----------------+------------------+--------------------+--------------------+-----------+-------+-----------+----------------+---------+
```
```bash
text
0	Patient Nam Financial Numbe Random Hospital Date of Birth Patient Location Chief Complaint Shortness of breath History of Present Illness Patient is an 84-year-old male wilh a past medical history of hypertension, HFpEF last known EF 55%, mild to moderate TA, pulmonary hypertension, permanent atrial fibrillation on Eliquis, history of GI blesd, CK-M8, and anemia who presents with full weeks oi ccneralized fatigue and fecling unwell. He also notes some shortness oi Breath and worsening dyspnea willy minimal exerlion. His major complaints are shoulder and joint pains. diffusely. He also complains of "bone pain’. He denics having any fevers or cnills. e demes having any chest pain, palpitalicns, He denies any worse extremity swelling than his baseline. He states he’s been compliant with his mcdications. Although he stales he ran out of his Eliquis & few weeks ago. He denies having any blood in his stools or mc!ena, although he does take iron pills and states his stools arc irequently black. His hemoglobin Is al baseline. Twelve-lead EKG showing atrial fibrillation, RBBB, LAFB, PVC. Chest x-ray showing new small right creater than left pleural effusions with mild pulmonary vascular congestion. BNP increased to 2800, up fram 1900. Tropoain 0.03. Renal function at baseline. Hemoaglopin at baseline. She normally takes 80 mq of oral Lasix daily. He was given 80 mg of IV Lasix in the ED. He is currently net negative close to 1 L. He is still on 2 L nasal cannula. ' Ss 5 A 10 system roview af systems was completed and negative except as documented in HPI. Physical Exam Vitals & Measurements T: 36.8 °C (Oral) TMIN: 36.8 "C (Oral) TMAX: 37.0 °C (Oral) HR: 54 RR: 7 BP: 140/63 WT: 100.3 KG Pulse Ox: 100 % Oxygen: 2 L'min via Nasal Cannula GENERAL: no acute distress HEAD: normecephalic EYES‘EARS‘NOSE/THAOAT: nupils are equal. normal oropharynx NECK: normal inspection RESPIRATORY: no respiratory distress, no rales on my exam CARDIOVASCULAR: irregular. brady. no murmurs, rubs or galleps ABDOMEN: soft, non-tendes EXTREMITIES: Bilateral chronic venous stasis changes NEUROLOGIC: alert and osieniec x 3. no gross motar or sensory deficils AssessmenvPlan Acute on chronic diastolic CHF (congestive heart failure) Acute on chronic diastolic heart failure exacerbation. Small pleural effusions dilaterally with mild pulmonary vascular congesiion on chest x-ray, slighi elevation in BNR. We'll continue 1 more day af IV diuresis with 80 mg IV Lasix. He may have had a viral infection which precipilated this. We'll add Tylenol jor his joint paias. Continue atenclol and chiorthalidone. AF - Atrial fibrillation Permanent atrial fibrillation. Rates bradycardic in the &0s. Continue atenolol with hola parameters. Coniinue Eliquis for stroke prevention. No evidence oj bleeding, hemog'abin at baseline. Printed: 7/17/2017 13:01 EDT Page 16 of 42 Arincitis CHF - Congestive heart failure Chronic kidney disease Chronic venous insufficiency Edema GI bleeding Glaucoma Goul Hypertension Peptic ulcer Peripheral ncuropathy Peripheral vascular disease Pulmonary hypertension Tricuspid regurgitation Historical No qualifying data Procedure/Surgical History duodenal resection, duodenojcjunostomy. small bowel enterolomy, removal of foreign object and repair oi enterotomy (05/2 1/20 14), colonoscopy (12/10/2013), egd (1209/2013), H/O endoscopy (07/2013), H’O colonoscopy (03/2013), pilonidal cyst removal at base of spine (1981), laser eye surgery ior glaucoma. lesions on small intestine closed up. Home Medications Home allopurinol 300 mg oral tablet, 300 MG= 1 TAB, PO. Daily atenolol 25 mg oral tablet, 25 MG= 1 TAB, PO, Daily chtorthalidone 25 mg oral tablet, 23 MG= 1 TAB, PO, MVE Combigan 0.2%-0.5% ophthalmic solution, 1 DROP, Both Eyes, Q12H Eliquis 5 mg oral lablet, 5 MG= 1 TAB, PO, BID lerrous sulfate 925 mg (65 nig elemental iron) oral tablet, 325 MG= 1 TAB, PO, Daily Lasix 80 mg oral tabic:. 80 MG= | TAB. PO, BID omeprazole 20 mg oral delayed scicasc capsule, 20 MG= 1 CAP, PO, BID Percocei 5/325 oral tablet. | TAB, PO. QAM potassium chloride 20 mEq oral tablet, extended release, 20 MEO= 1 TAB, PO, Daily sertraline 50 mg oral tablet, 75 MG= 1,5 TAB, PQ. Daiiy triamcinolone 0.71% lopical cream, 1 APP, Topical, Daily lriamcmnolone 0.1% lopical ominient, 1 APP. Topical, Daily PowerChart
1	Patient Name Financial Number Date of Girth Patient Location Random Hospital H & P Anemia Vitamin D2 50,000 intl units (1.25 ma) oral ALBASeRne capsule, 1 TAS, PO, Veexly-Tue Arthritis Allergies Tylenol for pain. Patient also takes Percocet alt home, will add this cn. Chronic kidney disease AY baseline. Monitor while divresing. Hypertension Blood pressures within tolerable ranges. Pulmonary hypertension Tricuspid regurgitation Mild-to-moderaie on echocardiogram last year sholliisn (cout) sulfa drug (maculopapular rash) Social History Ever Smoked tobacco: Former Smoker Alcohol use - frequency; None Drug use: Never Lab Results O7/16/9 7 05:30 to O7/16/17 05:30 Attending physician note-the patient was interviewed and examined. The appropriatc information in power chart was reviewed. The patient was discussed wilh Dr, Persad. 143 1L 981H 26? Patient may have @ mild degree of heart failure. He and his wife were more concernes with ee Ins peripheral edema. He has underlying renal insufficiency as well. We'll try to diurese him to his “dry" weight. We will then try to adjust his medications to kcep him within & narrow range of [hat weight. We will stop his atenolol this point since he is relatively bradycardic anc observe his heart rate on the cardiac monitor. He will progress with his care and aclivily as tolerated. 102 07/16/17 05:30 to O7/ 16/17 05:30 05:30 GLU 102 mg/dL Printed: 7/1 7/2017 13:01 EDT Page 17 of 42 NA K CL TOTAL COZ BUN CRT ANION GAP CA CBC with diff WBC HGB HCT RBC MCV MICH MCHC RDW MPV 143 MMOL/L 3.6 MMOL/L 98 MMOL/L 40 MMOL/L 26 mg/dL. 1.23 mg/dL 5 7.9 mg/dL 07/16/17 05:30 3.4/ nl 10.1 G/DL 32.4 %o 3.41 /PL 95.0 FL 29.6 pg 31.2 % 15,9 %o 10.7 FL PowerChart
2	Alexandria is the second-largest city in Egypt and a major economic centre, extending about 32 km (20 mi) along the coast of the Mediterranean Sea in the north central part of the country. Its low elevation on the Nile delta makes it highly vulnerable to rising sea levels. Alexandria is an important industrial center because of its natural gas and oil pipelines from Suez. Alexandria is also a popular tourist destination.
3	Alexandria was founded around a small, ancient Egyptian town c. 332 BC by Alexander the Great,[4] king of Macedon and leader of the Greek League of Corinth, during his conquest of the Achaemenid Empire. Alexandria became an important center of Hellenistic civilization and remained the capital of Ptolemaic Egypt and Roman and Byzantine Egypt for almost 1,000 years, until the Muslim conquest of Egypt in AD 641, when a new capital was founded at Fustat (later absorbed into Cairo). Hellenistic Alexandria was best known for the Lighthouse of Alexandria (Pharos), one of the Seven Wonders of the Ancient World; its Great Library (the largest in the ancient world); and the Necropolis, one of the Seven Wonders of the Middle Ages. Alexandria was at one time the second most powerful city of the ancient Mediterranean region, after Rome. Ongoing maritime archaeology in the harbor of Alexandria, which began in 1994, is revealing details of Alexandria both before the arrival of Alexander, when a city named Rhacotis existed there, and during the Ptolemaic dynasty.
4	8 i , . ! 9 i , . ! 10 i , . ! 11 i , . ! 12 i , . ! 13 i , . ! 14 i , . !```
