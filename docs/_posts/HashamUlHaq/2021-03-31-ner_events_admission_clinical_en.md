---
layout: model
title: Detect Clinical Events (Admissions)
author: John Snow Labs
name: ner_events_admission_clinical
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

This model can be used to detect clinical events in medical text, with a focus on admission entities.

## Predicted Entities

`DATE`, `TIME`, `PROBLEM`, `TEST`, `TREATMENT`, `OCCURENCE`, `CLINICAL_DEPT`, `EVIDENTIAL`, `DURATION`, `FREQUENCY`, `ADMISSION`, `DISCHARGE`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_admission_clinical_en_3.0.0_3.0_1617209704296.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
         
sentence_detector = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_events_admission_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["The patient presented to the emergency room last evening"]], ["text"]))
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_events_admission_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val data = Seq("""The patient presented to the emergency room last evening""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.admission_events").predict("""The patient presented to the emergency room last evening""")
```

</div>

## Results

```bash
+----+-----------------------------+---------+---------+-----------------+
|    | chunk                       |   begin |   end   |     entity      |
+====+=============================+=========+=========+=================+
|  0 | presented                   |    12   |    20   |   EVIDENTIAL    |
+----+-----------------------------+---------+---------+-----------------+
|  1 | the emergency room          |    25   |    42   |  CLINICAL_DEPT  |
+----+-----------------------------+---------+---------+-----------------+
|  2 | last evening                |    44   |    55   |     DATE        |
+----+-----------------------------+---------+---------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_events_admission_clinical|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Trained on augmented/enriched i2b2 events data with clinical_embeddings. The data for Admissions has been enriched specifically.

## Benchmarking

```bash
label	           tp	   fp	  fn	 prec	       rec	       f1
I-TIME	           42	   6	  9	    0.875	     0.8235294	 0.8484849
I-TREATMENT	       1134	   111	  312	0.9108434	 0.7842324	 0.8428094
B-OCCURRENCE	   406	   344	  382	0.5413333	 0.51522845	 0.52795845
I-DURATION	       160	   42	  71	0.7920792	 0.6926407	 0.73903
B-DATE	           500	   32	  49	0.9398496	 0.9107468	 0.92506933
I-DATE	           309	   54	  49	0.8512397	 0.8631285	 0.8571429
B-ADMISSION	       206	   1	  2	    0.9951691	 0.99038464	 0.9927711
I-PROBLEM	       2394	   390	  412	0.85991377	 0.85317177	 0.8565295
B-CLINICAL_DEPT	   327	   64	  77	0.8363171	 0.8094059	 0.8226415
B-TIME	           44	   12	  15	0.78571427	 0.7457627	 0.76521736
I-CLINICAL_DEPT	   597	   62	  78	0.90591806	 0.8844444	 0.8950525
B-PROBLEM	       1643	   260	  252	0.86337364	 0.86701846	 0.86519223
I-FREQUENCY	       35	   21	  39	0.625	     0.47297296	 0.5384615
I-TEST	           1082	   171	  117	0.86352754	 0.9024187	 0.8825449
B-TEST	           781	   125	  127	0.8620309	 0.86013216	 0.86108047
B-TREATMENT	       1283	   176	  202	0.87936944	 0.8639731	 0.87160325
B-DISCHARGE	       155	   0	  1	    1.0	         0.99358976	 0.99678457
B-EVIDENTIAL	   269	   25	  75	0.914966	 0.78197676	 0.84326017
B-DURATION	       97	   43	  44	0.69285715	 0.6879433	 0.6903914
B-FREQUENCY	       70	   16	  33	0.81395346	 0.6796116	 0.7407407

tp: 11841 fp: 2366 fn: 2680 labels: 22
Macro-average	 prec: 0.8137135, rec: 0.7533389, f1: 0.7823631
Micro-average	 prec: 0.83346236, rec: 0.8154397, f1: 0.8243525
```