---
layout: model
title: Detect clinical concepts (jsl_ner_wip_modifier_clinical)
author: John Snow Labs
name: jsl_ner_wip_modifier_clinical
date: 2021-04-01
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


Detect modifiers and other clinical entities using pretrained NER model.


## Predicted Entities


`Kidney_Disease`, `Height`, `Family_History_Header`, `RelativeTime`, `Hypertension`, `HDL`, `Alcohol`, `Test`, `Substance`, `Fetus_NewBorn`, `Diet`, `Substance_Quantity`, `Allergen`, `Form`, `Birth_Entity`, `Age`, `Race_Ethnicity`, `Modifier`, `Internal_organ_or_component`, `Hyperlipidemia`, `ImagingFindings`, `Psychological_Condition`, `Triglycerides`, `Cerebrovascular_Disease`, `Obesity`, `Duration`, `Weight`, `Date`, `Test_Result`, `Strength`, `VS_Finding`, `Respiration`, `Social_History_Header`, `Employment`, `Injury_or_Poisoning`, `Medical_History_Header`, `Death_Entity`, `Relationship_Status`, `Oxygen_Therapy`, `Blood_Pressure`, `Gender`, `Section_Header`, `Oncological`, `Drug`, `Labour_Delivery`, `Heart_Disease`, `LDL`, `Medical_Device`, `Temperature`, `Treatment`, `Female_Reproductive_Status`, `Total_Cholesterol`, `Time`, `Disease_Syndrome_Disorder`, `Communicable_Disease`, `EKG_Findings`, `Diabetes`, `Route`, `External_body_part_or_region`, `Pulse`, `Vital_Signs_Header`, `Direction`, `Admission_Discharge`, `Overweight`, `RelativeDate`, `O2_Saturation`, `BMI`, `Vaccine`, `Pregnancy`, `Sexually_Active_or_Sexual_Orientation`, `Procedure`, `Frequency`, `Dosage`, `Symptom`, `Clinical_Dept`, `Smoking`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_modifier_clinical_en_3.0.0_3.0_1617260799422.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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
		
jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_modifier_clinical", "en", "clinical/models") \
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

data = spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]]).toDF("text")

result = jsl_ner_model.transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
		.setInputCol("text")
		.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
		.setInputCols("document") 
		.setOutputCol("sentence")

val tokenizer = new Tokenizer()
		.setInputCols("sentence")
		.setOutputCol("token")
	
val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
		.setInputCols(Array("sentence",  "token")) 
		.setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("jsl_ner_wip_modifier_clinical", "en", "clinical/models")
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
nlu.load("en.med_ner.jsl.wip.clinical.modifier").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
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
|Model Name:|jsl_ner_wip_modifier_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg1OTY3OTY2NywzMjA0OTA4MDEsMTEzNz
A5Mzg4Ml19
-->
