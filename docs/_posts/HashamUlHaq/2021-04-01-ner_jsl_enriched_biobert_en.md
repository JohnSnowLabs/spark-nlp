---
layout: model
title: Detect clinical entities (ner_jsl_enriched_biobert)
author: John Snow Labs
name: ner_jsl_enriched_biobert
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


Detect symptoms, modifiers, age, drugs, treatments, tests and a lot more using a single pretrained NER model.


## Predicted Entities


`Symptom_Name`, `Pulse_Rate`, `Negation`, `Age`, `Modifier`, `Substance_Name`, `Causative_Agents_(Virus_and_Bacteria)`, `Diagnosis`, `Weight`, `Drug_Name`, `Procedure_Name`, `Lab_Name`, `Blood_Pressure`, `Lab_Result`, `Gender`, `Name`, `Temperature`, `Section_Name`, `Route`, `Maybe`, `O2_Saturation`, `Respiratory_Rate`, `Procedure`, `Frequency`, `Dosage`, `Allergenic_substance`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_JSL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_JSL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_biobert_en_3.0.0_3.0_1617260842011.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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
	
embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")\
		.setInputCols(["sentence",  "token"]) \
		.setOutputCol("embeddings")
		
jsl_ner = MedicalNerModel.pretrained("ner_jsl_enriched_biobert", "en", "clinical/models") \
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

model = jsl_ner_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."]], ["text"]))
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
	
val embeddings = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
		.setInputCols(Array("sentence", "token")) 
		.setOutputCol("embeddings")

val jsl_ner = MedicalNerModel.pretrained("ner_jsl_enriched_biobert", "en", "clinical/models")
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
nlu.load("en.med_ner.jsl.enriched_biobert").predict("""The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.""")
```

</div>

## Results

```bash
+---------------------------+------------+
|chunk                      |ner_label   |
+---------------------------+------------+
|21-day-old                 |Age         |
|male                       |Gender      |
|mom                        |Gender      |
|she                        |Gender      |
|mild                       |Modifier    |
|problems with his breathing|Symptom_Name|
|negative                   |Negation    |
|perioral cyanosis          |Symptom_Name|
|retractions                |Symptom_Name|
|mom                        |Gender      |
|Tylenol                    |Drug_Name   |
|His                        |Gender      |
|his                        |Gender      |
|respiratory congestion     |Symptom_Name|
|He                         |Gender      |
|tired                      |Symptom_Name|
|fussy                      |Symptom_Name|
|albuterol                  |Drug_Name   |
|His                        |Gender      |
|he                         |Gender      |
|he                         |Gender      |
|Mom                        |Gender      |
|denies                     |Negation    |
|diarrhea                   |Symptom_Name|
|His                        |Gender      |
+---------------------------+------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_jsl_enriched_biobert|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg4MDc3MzE1MywtOTY1NDAzNTc3LDEzMj
g3NTgxNjQsMTA2NzI2MjA2NCwtOTQxMTc4NDcyLDE5Mjc2ODgx
NjQsLTc3NzkwMzU4MCwyMDcyOTU2MzQ2XX0=
-->
