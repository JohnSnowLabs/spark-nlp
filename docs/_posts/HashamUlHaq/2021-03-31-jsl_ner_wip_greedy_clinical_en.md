---
layout: model
title: Detect Clinical Entities (jsl_ner_wip_greedy_clinical)
author: John Snow Labs
name: jsl_ner_wip_greedy_clinical
date: 2021-03-31
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for clinical terminology. The SparkNLP deep learning model (MedicalNerModel) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.


## Predicted Entities


`Injury_or_Poisoning`, `Direction`, `Test`, `Admission_Discharge`, `Death_Entity`, `Relationship_Status`, `Duration`, `Hyperlipidemia`, `Respiration`, `Birth_Entity`, `Age`, `Family_History_Header`, `Labour_Delivery`, `BMI`, `Temperature`, `Alcohol`, `Kidney_Disease`, `Oncological`, `Medical_History_Header`, `Cerebrovascular_Disease`, `Oxygen_Therapy`, `O2_Saturation`, `Psychological_Condition`, `Heart_Disease`, `Employment`, `Obesity`, `Disease_Syndrome_Disorder`, `Pregnancy`, `ImagingFindings`, `Procedure`, `Medical_Device`, `Race_Ethnicity`, `Section_Header`, `Drug`, `Symptom`, `Treatment`, `Substance`, `Route`, `Blood_Pressure`, `Diet`, `External_body_part_or_region`, `LDL`, `VS_Finding`, `Allergen`, `EKG_Findings`, `Imaging_Technique`, `Triglycerides`, `RelativeTime`, `Gender`, `Pulse`, `Social_History_Header`, `Substance_Quantity`, `Diabetes`, `Modifier`, `Internal_organ_or_component`, `Clinical_Dept`, `Form`, `Strength`, `Fetus_NewBorn`, `RelativeDate`, `Height`, `Test_Result`, `Time`, `Frequency`, `Sexually_Active_or_Sexual_Orientation`, `Weight`, `Vaccine`, `Vital_Signs_Header`, `Communicable_Disease`, `Dosage`, `Hypertension`, `HDL`, `Overweight`, `Total_Cholesterol`, `Smoking`, `Date`.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_greedy_clinical_en_3.0.0_3.0_1617206898504.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
		.setInputCols(["document"]) \
		.setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")
	
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
		.setInputCols(["sentence",  "token"])\
		.setOutputCol("embeddings")
		
jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("jsl_ner")

jsl_ner_converter = NerConverter() \
		.setInputCols(["sentence", "token", "jsl_ner"]) \
		.setOutputCol("ner_chunk")

jsl_ner_pipeline = Pipeline().setStages([
				documentAssembler,
				sentenceDetector,
				tokenizer,
				embeddings,
				jsl_ner,
				jsl_ner_converter])


jsl_ner_model = jsl_ner_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."""]]).toDF("text")

result = jsl_ner_model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
		.setInputCols("document") 
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols("sentence")
		.setOutputCol("token")
	
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en",  "clinical/models")
		.setInputCols(Array("sentence",  "token")) 
		.setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_greedy_clinical", "en", "clinical/models")
		.setInputCols(Array("sentence", "token", "embeddings"))
		.setOutputCol("jsl_ner")

val jsl_ner_converter = new NerConverter()
		.setInputCols(Array("sentence", "token", "jsl_ner"))
		.setOutputCol("ner_chunk")

val jsl_ner_pipeline = new Pipeline().setStages(Array(
					documentAssembler, 
					sentenceDetector, 
					tokenizer, 
					embeddings, 
					jsl_ner, 
					jsl_ner_converter))


val data = Seq("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""").toDS.toDF("text")

val result = jsl_ner_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.jsl.wip.clinical.greedy").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```

</div>


## Results


```bash
+----------------------------------------------+----------------------------+
|chunk                                         |ner_label                   |
+----------------------------------------------+----------------------------+
|21-day-old                                    |Age                         |
|Caucasian                                     |Race_Ethnicity              |
|male                                          |Gender                      |
|for 2 days                                    |Duration                    |
|congestion                                    |Symptom                     |
|mom                                           |Gender                      |
|suctioning yellow discharge                   |Symptom                     |
|nares                                         |External_body_part_or_region|
|she                                           |Gender                      |
|mild problems with his breathing while feeding|Symptom                     |
|perioral cyanosis                             |Symptom                     |
|retractions                                   |Symptom                     |
|One day ago                                   |RelativeDate                |
|mom                                           |Gender                      |
|tactile temperature                           |Symptom                     |
|Tylenol                                       |Drug                        |
|Baby                                          |Age                         |
|decreased p.o. intake                         |Symptom                     |
|His                                           |Gender                      |
|20 minutes                                    |Duration                    |
|q.2h.                                         |Frequency                   |
|to 5 to 10 minutes                            |Duration                    |
|his                                           |Gender                      |
|respiratory congestion                        |Symptom                     |
|He                                            |Gender                      |
|tired                                         |Symptom                     |
|fussy                                         |Symptom                     |
|over the past 2 days                          |RelativeDate                |
|albuterol                                     |Drug                        |
|ER                                            |Clinical_Dept               |
|His                                           |Gender                      |
|urine output has also decreased               |Symptom                     |
|he                                            |Gender                      |
|per 24 hours                                  |Frequency                   |
|he                                            |Gender                      |
|per 24 hours                                  |Frequency                   |
|Mom                                           |Gender                      |
|diarrhea                                      |Symptom                     |
|His                                           |Gender                      |
|bowel                                         |Internal_organ_or_component |
+----------------------------------------------+----------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_greedy_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Data Source


Trained on data gathered and manually annotated by John Snow Labs.
https://www.johnsnowlabs.com/data/


## Benchmarking


```bash
entity					tp      fp      fn		total  precision  recall   f1
VS_Finding				229.0	56.0    34.0	263.0	0.8035  0.8707  0.8358
Direction				4009.0	479.0   403.0	4412.0	0.8933  0.9087  0.9009
Female_Reproducti...	2.0     1.0     3.0     5.0     0.6667  0.4     0.5
Respiration				80.0	9.0		14.0    94.0	0.8989  0.8511  0.8743
Cerebrovascular_D...	82.0    27.0    18.0	100.0	0.7523  0.82  	0.7847
not						4.0     0.0     0.0     4.0		1.0     1.0     1.0
Family_History_He...    86.0	4.0     3.0		89.0	0.9556  0.9663  0.9609
Heart_Disease			469.0	76.0    83.0	552.0	0.8606  0.8496  0.8551
ImagingFindings			68.0    38.0    75.0	143.0	0.6415  0.4755  0.5462
RelativeTime			141.0	76.0    66.0	207.0	0.6498  0.6812  0.6651
Strength				720.0	49.0    58.0	778.0	0.9363  0.9254  0.9308
Smoking					117.0	8.0     6.0		123.0	0.936  	0.9512  0.9435
Medical_Device			3584.0	730.0   359.0	3943.0	0.8308  0.909  	0.8681
EKG_Findings			41.0    20.0    45.0    86.0	0.6721  0.4767  0.5578
Pulse					138.0	23.0    24.0	162.0	0.8571  0.8519  0.8545
Psychological_Con...	121.0	14.0    29.0	150.0	0.8963  0.8067  0.8491
Overweight				5.0     2.0     0.0     5.0     0.7143  1.0  	0.8333
Triglycerides			3.0     0.0     0.0     3.0		1.0     1.0     1.0
Obesity					49.0	6.0     4.0		53.0	0.8909  0.9245  0.9074
Admission_Discharge		325.0	30.0	2.0		327.0	0.9155  0.9939  0.9531
HDL						2.0		1.0     1.0     3.0     0.6667  0.6667  0.6667
Diabetes				118.0	13.0	7.0		125.0	0.9008  0.944  	0.9219
Section_Header			3778.0	148.0   138.0	3916.0	0.9623  0.9648  0.9635
Age						617.0	52.0    47.0	664.0	0.9223  0.9292  0.9257
O2_Saturation			34.0    11.0    19.0    53.0	0.7556  0.6415  0.6939
Kidney_Disease			114.0	5.0		12.0	126.0	0.958  	0.9048  0.9306
Test					2668.0	526.0   498.0	3166.0	0.8353  0.8427  0.839
Communicable_Disease	25.0    12.0	9.0		34.0	0.6757  0.7353  0.7042
Hypertension			152.0	10.0	6.0		158.0	0.9383   0.962  0.95
External_body_par...	652.0   387.0   340.0	2992.0	0.8727  0.8864  0.8795
Oxygen_Therapy			67.0    21.0    23.0    90.0     0.7614  0.7444 0.7528
Test_Result				1124.0	227.0   258.0	1382.0	0.832  	0.8133  0.8225
Modifier				539.0   185.0   309.0   848.0	0.7445  0.6356  0.6858
BMI						7.0     1.0     1.0     8.0		0.875   0.875   0.875
Labour_Delivery			75.0    19.0    23.0    98.0	0.7979  0.7653  0.7813
Employment				249.0	51.0    57.0	306.0	0.83  	0.8137  0.8218
Clinical_Dept			948.0	95.0    80.0	1028.0	0.9089  0.9222  0.9155
Time					36.0	7.0     7.0		43.0	0.8372  0.8372  0.8372
Procedure				3180.0	460.0   480.0	3660.0	0.8736  0.8689  0.8712
Diet					50.0	29.0    30.0    80.0	0.6329   0.625  0.6289
Oncological				478.0	46.0    50.0	528.0	0.9122  0.9053  0.9087
LDL						3.0		0.0     2.0     5.0		1.0     0.6    	0.75
Symptom					6801.0	1097.0  1097.0  7898.0	0.8611  0.8611  0.8611
Temperature				109.0	12.0	7.0		116.0	0.9008  0.9397  0.9198
Vital_Signs_Header		213.0	27.0    16.0	229.0	0.8875  0.9301  0.9083
Relationship_Status		42.0	2.0     1.0		43.0	0.9545  0.9767  0.9655
Total_Cholesterol		10.0	4.0     5.0		15.0	0.7143  0.6667  0.6897
Blood_Pressure			167.0	22.0    23.0	190.0	0.8836  0.8789  0.8813
Injury_or_Poisoning		510.0	83.0	111.0   621.0	0.86  	0.8213  0.8402
Drug_Ingredient			1698.0	160.0   158.0	1856.0	0.9139  0.9149  0.9144
Treatment				156.0	40.0    54.0	210.0	0.7959  0.7429  0.7685
Assertion_SocialD...	4.0		0.0     6.0		10.0	1.0     0.4		0.5714
Pregnancy				100.0	45.0    41.0	141.0	0.6897  0.7092  0.6993
Vaccine					13.0	3.0     6.0		19.0	0.8125  0.6842  0.7429
Disease_Syndrome_...	2861.0	452.0   376.0	3237.0	0.8636  0.8838  0.8736
Height					25.0	8.0     9.0		34.0	0.7576  0.7353  0.7463
Frequency				650.0	157.0   148.0   798.0	0.8055  0.8145	0.81
Route					872.0	83.0    85.0	957.0	0.9131  0.9112  0.9121
Death_Entity			49.0	7.0     6.0		55.0	0.875 	 0.8909	0.8829
Duration				367.0   132.0	95.0	462.0	0.7355  0.7944  0.7638
Internal_organ_or...	6532.0  1016.0	987.0	7519.0	0.8654  0.8687  0.8671
Alcohol					79.0    20.0    12.0    91.0	0.798 	 0.8681	0.8316
Date					515.0	19.0    19.0	534.0	0.9644  0.9644  0.9644
Hyperlipidemia			47.0	2.0     1.0		48.0	0.9592  0.9792  0.9691
Social_History_He...	89.0	9.0     4.0		93.0	0.9082   0.957  0.9319
Race_Ethnicity			113.0 	0.0     3.0		116.0	1.0		0.9741  0.9869
Imaging_Technique		47.0    31.0    30.0    77.0	0.6026  0.6104  0.6065
Drug_BrandName			963.0	72.0    79.0	1042.0	0.9304  0.9242  0.9273
RelativeDate			553.0   128.0   121.0   674.0	0.812   0.8205  0.8162
Gender					6043.0	59.0    87.0	6130.0	0.9903  0.9858  0.9881
Form					227.0	35.0    47.0	274.0	0.8664  0.8285	0.847
Dosage					279.0	42.0    62.0	341.0	0.8692  0.8182  0.8429
Medical_History_H...	117.0	4.0		11.0	128.0	0.9669  0.9141  0.9398
Substance				59.0    16.0    16.0    75.0	0.7867  0.7867  0.7867
Weight					85.0    19.0    21.0	106.0	0.8173  0.8019  0.8095
macro					-       -       -       -		-       -		0.7286
micro					-       -       -       -		-       -		0.8715
```