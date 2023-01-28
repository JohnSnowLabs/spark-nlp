---
layout: model
title: Detect Clinical Entities (bert_token_classifier_ner_jsl)
author: John Snow Labs
name: bert_token_classifier_ner_jsl
date: 2021-08-28
tags: [ner, en, licensed, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.2.0
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is BERT-based version of `ner_jsl` model and it is better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture.


## Predicted Entities


`Injury_or_Poisoning`, `Direction`, `Test`, `Admission_Discharge`, `Death_Entity`, `Relationship_Status`, `Duration`, `Respiration`, `Hyperlipidemia`, `Birth_Entity`, `Age`, `Labour_Delivery`, `Family_History_Header`, `BMI`, `Temperature`, `Alcohol`, `Kidney_Disease`, `Oncological`, `Medical_History_Header`, `Cerebrovascular_Disease`, `Oxygen_Therapy`, `O2_Saturation`, `Psychological_Condition`, `Heart_Disease`, `Employment`, `Obesity`, `Disease_Syndrome_Disorder`, `Pregnancy`, `ImagingFindings`, `Procedure`, `Medical_Device`, `Race_Ethnicity`, `Section_Header`, `Symptom`, `Treatment`, `Substance`, `Route`, `Drug_Ingredient`, `Blood_Pressure`, `Diet`, `External_body_part_or_region`, `LDL`, `VS_Finding`, `Allergen`, `EKG_Findings`, `Imaging_Technique`,  `Triglycerides`, `RelativeTime`, `Gender`, `Pulse`, `Social_History_Header`, `Substance_Quantity`, `Diabetes`, `Modifier`, `Internal_organ_or_component`, `Clinical_Dept`, `Form`, `Drug_BrandName`, `Strength`, `Fetus_NewBorn`, `RelativeDate`, `Height`, `Test_Result`, `Sexually_Active_or_Sexual_Orientation`, `Frequency`, `Time`, `Weight`, `Vaccine`, `Vital_Signs_Header`, `Communicable_Disease`, `Dosage`, `Overweight`, `Hypertension`, `HDL`, `Total_Cholesterol`, `Smoking`, `Date`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_BERT_TOKEN_CLASSIFIER/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_BERT_TOKEN_CLASSIFIER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_en_3.2.0_2.4_1630172634235.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_en_3.2.0_2.4_1630172634235.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jsl", "en", "clinical/models")\
.setInputCols("token", "sentence")\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter()\
.setInputCols(["sentence","token","ner"])\
.setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
	       documentAssembler,
	       sentenceDetector,
	       tokenizer,
	       tokenClassifier,
	       ner_converter])
						       
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

sample_text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge ."""

result = model.transform(spark.createDataFrame([[sample_text]]).toDF("text"))
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
		.setInputCols(Array("document"))
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols("sentence")
		.setOutputCol("token")
		
val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_jsl", "en", "clinical/models")
		.setInputCols(Array("token", "sentence"))
		.setOutputCol("ner")
		.setCaseSensitive(True)

val ner_converter = new NerConverter()
		.setInputCols(Array("sentence","token","ner"))
		.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(
				documentAssembler,
				sentenceDetector,
				tokenizer,
				tokenClassifier,
				ner_converter))
												
val sample_text = Seq("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .""").toDS.toDF("text")

val result = pipeline.fit(sample_text).transform(sample_text)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_jsl").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .""")
```

</div>


## Results


```bash
+------------+-------------------------+
|chunk       |label                    |
+------------+-------------------------+
|28-year-old |Age                      |
|female      |Gender                   |
|gestational |Diabetes                 |
|diabetes    |Diabetes                 |
|mellitus    |Diabetes                 |
|eight       |RelativeDate             |
|years       |RelativeDate             |
|prior       |RelativeDate             |
|type        |Diabetes                 |
|two         |Diabetes                 |
|diabetes    |Diabetes                 |
|mellitus    |Diabetes                 |
|T2DM        |Diabetes                 |
|HTG-induced |Diabetes                 |
|pancreatitis|Disease_Syndrome_Disorder|
|three       |RelativeDate             |
|years       |RelativeDate             |
|prior       |RelativeDate             |
|acute       |Disease_Syndrome_Disorder|
|hepatitis   |Disease_Syndrome_Disorder|
|obesity     |Obesity                  |
|body        |BMI                      |
|mass        |BMI                      |
|index       |BMI                      |
|BMI         |BMI                      |
|)           |BMI                      |
|of          |BMI                      |
|33.5        |BMI                      |
|kg/m2       |BMI                      |
|polyuria    |Symptom                  |
|polydipsia  |Symptom                  |
|poor        |Symptom                  |
|appetite    |Symptom                  |
|vomiting    |Symptom                  |
|Two         |RelativeDate             |
|weeks       |RelativeDate             |
|prior       |RelativeDate             |
|she         |Gender                   |
|five-day    |Drug                     |
|course      |Drug                     |
+------------+-------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl|
|Compatibility:|Healthcare NLP 3.2.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|128|


## Data Source


Trained on data gathered and manually annotated by John Snow Labs. https://www.johnsnowlabs.com/data/


## Benchmarking


```bash
label       precision recall   f1-score  support
Admission_Discharge       0.84      0.97      0.90       415
Age       0.96      0.96      0.96      2434
Alcohol       0.75      0.83      0.79       145
BMI       1.00      0.77      0.87        26
Blood_Pressure       0.86      0.88      0.87       597
Cerebrovascular_Disease       0.74      0.77      0.75       266
Clinical_Dept       0.90      0.92      0.91      2385
Communicable_Disease       0.70      0.59      0.64        85
Date       0.95      0.98      0.96      1438
Death_Entity       0.83      0.83      0.83        59
Diabetes       0.95      0.95      0.95       350
Diet       0.60      0.49      0.54       229
Direction       0.88      0.90      0.89      6187
Disease_Syndrome_Disorder       0.90      0.89      0.89     13236
Dosage       0.57      0.49      0.53       263
Drug       0.91      0.93      0.92     15926
Duration       0.82      0.85      0.83      1218
EKG_Findings       0.64      0.70      0.67       325
Employment       0.79      0.85      0.82       539
External_body_part_or_region       0.84      0.84      0.84      4805
Family_History_Header       1.00      1.00      1.00       889
Fetus_NewBorn       0.57      0.56      0.56       341
Frequency       0.87      0.90      0.88      1718
Gender       0.98      0.98      0.98      5666
HDL       0.60      1.00      0.75         6
Heart_Disease       0.88      0.88      0.88      2295
Height       0.89      0.96      0.92       134
Hyperlipidemia       1.00      0.95      0.97       194
Hypertension       0.95      0.98      0.97       566
ImagingFindings       0.66      0.64      0.65       601
Imaging_Technique       0.62      0.67      0.64       108
Injury_or_Poisoning       0.85      0.83      0.84      1680
Internal_organ_or_component       0.90      0.91      0.90     21318
Kidney_Disease       0.89      0.89      0.89       446
LDL       0.88      0.97      0.92        37
Labour_Delivery       0.82      0.71      0.76       306
Medical_Device       0.89      0.93      0.91     12852
Medical_History_Header       0.96      0.97      0.96      1013
Modifier       0.68      0.60      0.64      1398
O2_Saturation       0.84      0.82      0.83       199
Obesity       0.96      0.98      0.97       130
Oncological       0.88      0.96      0.92      1635
Overweight       0.80      0.80      0.80        10
Oxygen_Therapy       0.91      0.92      0.92       231
Pregnancy       0.81      0.83      0.82       439
Procedure       0.91      0.91      0.91     14410
Psychological_Condition       0.81      0.81      0.81       354
Pulse       0.85      0.95      0.89       389
Race_Ethnicity       1.00      1.00      1.00       163
Relationship_Status       0.93      0.91      0.92        57
RelativeDate       0.83      0.86      0.84      1562
RelativeTime       0.74      0.79      0.77       431
Respiration       0.99      0.95      0.97       221
Route       0.68      0.69      0.69       597
Section_Header       0.97      0.98      0.98     28580
Sexually_Active_or_Sexual_Orientation       1.00      0.64      0.78        14
Smoking       0.83      0.90      0.86       225
Social_History_Header       0.95      0.99      0.97       825
Strength       0.71      0.55      0.62       227
Substance       0.85      0.81      0.83       193
Symptom       0.84      0.86      0.85     23092
Temperature       0.94      0.97      0.96       410
Test       0.84      0.88      0.86      9050
Test_Result       0.84      0.84      0.84      2766
Time       0.90      0.81      0.86       140
Total_Cholesterol       0.69      0.95      0.80        73
Treatment       0.73      0.72      0.73       506
Triglycerides       0.83      0.80      0.81        30
VS_Finding       0.76      0.77      0.76       588
Vaccine       0.70      0.84      0.76        92
Vital_Signs_Header       0.95      0.98      0.97      2223
Weight       0.88      0.89      0.88       306
O       0.97      0.96      0.97    253164
accuracy         -         -       0.94    445974
macro-avg       0.82      0.82      0.81    445974
weighted-avg       0.94      0.94      0.94    445974
```

