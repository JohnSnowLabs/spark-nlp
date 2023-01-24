---
layout: model
title: Detect Clinical Entities (jsl_ner_wip_clinical)
author: John Snow Labs
name: jsl_ner_wip_clinical
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


`Injury_or_Poisoning`, `Direction`, `Test`, `Admission_Discharge`, `Death_Entity`, `Relationship_Status`, `Duration`, `Respiration`, `Hyperlipidemia`, `Birth_Entity`, `Age`, `Labour_Delivery`, `Family_History_Header`, `BMI`, `Temperature`, `Alcohol`, `Kidney_Disease`, `Oncological`, `Medical_History_Header`, `Cerebrovascular_Disease`, `Oxygen_Therapy`, `O2_Saturation`, `Psychological_Condition`, `Heart_Disease`, `Employment`, `Obesity`, `Disease_Syndrome_Disorder`, `Pregnancy`, `ImagingFindings`, `Procedure`, `Medical_Device`, `Race_Ethnicity`, `Section_Header`, `Symptom`, `Treatment`, `Substance`, `Route`, `Drug_Ingredient`, `Blood_Pressure`, `Diet`, `External_body_part_or_region`, `LDL`, `VS_Finding`, `Allergen`, `EKG_Findings`, `Imaging_Technique`, `Triglycerides`, `RelativeTime`, `Gender`, `Pulse`, `Social_History_Header`, `Substance_Quantity`, `Diabetes`, `Modifier`, `Internal_organ_or_component`, `Clinical_Dept`, `Form`, `Drug_BrandName`, `Strength`, `Fetus_NewBorn`, `RelativeDate`, `Height`, `Test_Result`, `Sexually_Active_or_Sexual_Orientation`, `Frequency`, `Time`, `Weight`, `Vaccine`, `Vital_Signs_Header`, `Communicable_Disease`, `Dosage`, `Overweight`, `Hypertension`, `HDL`, `Total_Cholesterol`, `Smoking`, `Date`.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_clinical_en_3.0.0_3.0_1617208406089.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_clinical_en_3.0.0_3.0_1617208406089.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
		.setInputCol("text")\
		.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
		.setInputCols(["document"]) \
		.setOutputCol("sentence")

tokenizer = Tokenizer()\
		.setInputCols(["sentence"])\
		.setOutputCol("token")
	
embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
		.setInputCols(["sentence", "token"])\
		.setOutputCol("embeddings")

jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models") \
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


jsl_ner_model = jsl_ner_pipeline.fit(spark.createDataFrame([['']]).toDF("text"))

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
	
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
		.setInputCols(Array("sentence", "token"))
	    .setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_clinical", "en", "clinical/models")
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
nlu.load("en.med_ner").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```

</div>


## Results


```bash
+-----------------------------------------+----------------------------+
|chunk                                    |ner_label                   |
+-----------------------------------------+----------------------------+
|21-day-old                               |Age                         |
|Caucasian                                |Race_Ethnicity              |
|male                                     |Gender                      |
|for 2 days                               |Duration                    |
|congestion                               |Symptom                     |
|mom                                      |Gender                      |
|yellow                                   |Modifier                    |
|discharge                                |Symptom                     |
|nares                                    |External_body_part_or_region|
|she                                      |Gender                      |
|mild                                     |Modifier                    |
|problems with his breathing while feeding|Symptom                     |
|perioral cyanosis                        |Symptom                     |
|retractions                              |Symptom                     |
|One day ago                              |RelativeDate                |
|mom                                      |Gender                      |
|Tylenol                                  |Drug_BrandName              |
|Baby                                     |Age                         |
|decreased p.o. intake                    |Symptom                     |
|His                                      |Gender                      |
+-----------------------------------------+----------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_clinical|
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
entity      tp      fp      fn   total  precision  recall      f1
VS_Finding   235.0    46.0    43.0   278.0     0.8363  0.8453  0.8408
Direction  3972.0   465.0   458.0  4430.0     0.8952  0.8966  0.8959
Respiration    82.0     4.0     4.0    86.0     0.9535  0.9535  0.9535
Cerebrovascular_D...    93.0    20.0    24.0   117.0      0.823  0.7949  0.8087
Family_History_He...    88.0     6.0     3.0    91.0     0.9362   0.967  0.9514
Heart_Disease   447.0    82.0   119.0   566.0      0.845  0.7898  0.8164
RelativeTime   158.0    80.0    59.0   217.0     0.6639  0.7281  0.6945
Strength   624.0    58.0    53.0   677.0      0.915  0.9217  0.9183
Smoking   121.0    11.0     4.0   125.0     0.9167   0.968  0.9416
Medical_Device  3716.0   491.0   466.0  4182.0     0.8833  0.8886  0.8859
Pulse   136.0    22.0    14.0   150.0     0.8608  0.9067  0.8831
Psychological_Con...   135.0     9.0    29.0   164.0     0.9375  0.8232  0.8766
Overweight     2.0     1.0     0.0     2.0     0.6667     1.0     0.8
Triglycerides     3.0     0.0     2.0     5.0        1.0     0.6    0.75
Obesity    42.0     5.0     6.0    48.0     0.8936   0.875  0.8842
Admission_Discharge   318.0    24.0    11.0   329.0     0.9298  0.9666  0.9478
HDL     3.0     0.0     0.0     3.0        1.0     1.0     1.0
Diabetes   110.0    14.0     8.0   118.0     0.8871  0.9322  0.9091
Section_Header  3740.0   148.0   157.0  3897.0     0.9619  0.9597  0.9608
Age   627.0    75.0    48.0   675.0     0.8932  0.9289  0.9107
O2_Saturation    34.0    14.0    17.0    51.0     0.7083  0.6667  0.6869
Kidney_Disease    96.0    12.0    34.0   130.0     0.8889  0.7385  0.8067
Test  2504.0   545.0   498.0  3002.0     0.8213  0.8341  0.8276
Communicable_Disease    21.0    10.0     6.0    27.0     0.6774  0.7778  0.7241
Hypertension   162.0     5.0    10.0   172.0     0.9701  0.9419  0.9558
External_body_par...  2626.0   356.0   413.0  3039.0     0.8806  0.8641  0.8723
Oxygen_Therapy    81.0    15.0    14.0    95.0     0.8438  0.8526  0.8482
Modifier  2341.0   404.0   539.0  2880.0     0.8528  0.8128  0.8324
Test_Result  1007.0   214.0   255.0  1262.0     0.8247  0.7979  0.8111
BMI     9.0     1.0     0.0     9.0        0.9     1.0  0.9474
Labour_Delivery    57.0    23.0    33.0    90.0     0.7125  0.6333  0.6706
Employment   271.0    59.0    55.0   326.0     0.8212  0.8313  0.8262
Fetus_NewBorn    66.0    33.0    51.0   117.0     0.6667  0.5641  0.6111
Clinical_Dept   923.0   110.0    83.0  1006.0     0.8935  0.9175  0.9053
Time    29.0    13.0    16.0    45.0     0.6905  0.6444  0.6667
Procedure  3185.0   462.0   501.0  3686.0     0.8733  0.8641  0.8687
Diet    36.0    20.0    45.0    81.0     0.6429  0.4444  0.5255
Oncological   459.0    61.0    55.0   514.0     0.8827   0.893  0.8878
LDL     3.0     0.0     3.0     6.0        1.0     0.5  0.6667
Symptom  7104.0  1302.0  1200.0  8304.0     0.8451  0.8555  0.8503
Temperature   116.0     6.0     8.0   124.0     0.9508  0.9355  0.9431
Vital_Signs_Header   215.0    29.0    24.0   239.0     0.8811  0.8996  0.8903
Relationship_Status    49.0     2.0     1.0    50.0     0.9608    0.98  0.9703
Total_Cholesterol    11.0     4.0     5.0    16.0     0.7333  0.6875  0.7097
Blood_Pressure   158.0    18.0    22.0   180.0     0.8977  0.8778  0.8876
Injury_or_Poisoning   579.0   130.0   127.0   706.0     0.8166  0.8201  0.8184
Drug_Ingredient  1716.0   153.0   132.0  1848.0     0.9181  0.9286  0.9233
Treatment   136.0    36.0    60.0   196.0     0.7907  0.6939  0.7391
Pregnancy   123.0    36.0    51.0   174.0     0.7736  0.7069  0.7387
Vaccine    13.0     2.0     6.0    19.0     0.8667  0.6842  0.7647
Disease_Syndrome_...  2981.0   559.0   446.0  3427.0     0.8421  0.8699  0.8557
Height    30.0    10.0    15.0    45.0       0.75  0.6667  0.7059
Frequency   595.0    99.0   138.0   733.0     0.8573  0.8117  0.8339
Route   858.0    76.0    89.0   947.0     0.9186   0.906  0.9123
Duration   351.0    99.0   108.0   459.0       0.78  0.7647  0.7723
Death_Entity    43.0    14.0     5.0    48.0     0.7544  0.8958   0.819
Internal_organ_or...  6477.0   972.0   991.0  7468.0     0.8695  0.8673  0.8684
Alcohol    80.0    18.0    13.0    93.0     0.8163  0.8602  0.8377
Substance_Quantity     6.0     7.0     4.0    10.0     0.4615     0.6  0.5217
Date   498.0    38.0    19.0   517.0     0.9291  0.9632  0.9459
Hyperlipidemia    47.0     3.0     3.0    50.0       0.94    0.94    0.94
Social_History_He...    99.0     7.0     7.0   106.0      0.934   0.934   0.934
Race_Ethnicity   116.0     0.0     0.0   116.0        1.0     1.0     1.0
Imaging_Technique    40.0    18.0    47.0    87.0     0.6897  0.4598  0.5517
Drug_BrandName   859.0    62.0    61.0   920.0     0.9327  0.9337  0.9332
RelativeDate   566.0   124.0   143.0   709.0     0.8203  0.7983  0.8091
Gender  6096.0    80.0   101.0  6197.0      0.987  0.9837  0.9854
Dosage   244.0    31.0    57.0   301.0     0.8873  0.8106  0.8472
Form   234.0    32.0    55.0   289.0     0.8797  0.8097  0.8432
Medical_History_H...   114.0     9.0    10.0   124.0     0.9268  0.9194  0.9231
Birth_Entity     4.0     2.0     3.0     7.0     0.6667  0.5714  0.6154
Substance    59.0     8.0    11.0    70.0     0.8806  0.8429  0.8613
Sexually_Active_o...     5.0     3.0     4.0     9.0      0.625  0.5556  0.5882
Weight    90.0    10.0    21.0   111.0        0.9  0.8108  0.8531
macro     -       -       -       -         -       -     0.8148
micro     -       -       -       -         -       -     0.8788
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA0NDMwNDQ1MSwtNTk4OTY4NDA1LDM0Nj
QxOTk0MywtOTQ3MTExMTMyXX0=
-->
