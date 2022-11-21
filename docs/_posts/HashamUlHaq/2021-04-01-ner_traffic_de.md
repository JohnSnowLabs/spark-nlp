---
layout: model
title: Detect entities related to road traffic
author: John Snow Labs
name: ner_traffic
date: 2021-04-01
tags: [ner, clinical, licensed, de]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Detect entities related to road traffic using pretrained NER model.


## Predicted Entities


`ORGANIZATION_COMPANY`, `DISASTER_TYPE`, `TIME`, `TRIGGER`, `DATE`, `PERSON`, `LOCATION_STOP`, `ORGANIZATION`, `DISTANCE`, `LOCATION_STREET`, `NUMBER`, `DURATION`, `ORG_POSITION`, `LOCATION_ROUTE`, `LOCATION`, `LOCATION_CITY`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TRAFFIC_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_traffic_de_3.0.0_3.0_1617260858901.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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

embeddings_german = WordEmbeddingsModel.pretrained("w2v_cc_300d", "de", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_traffic", "de", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter()\
 	.setInputCols(["sentence", "token", "ner"])\
 	.setOutputCol("ner_chunk")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_german, clinical_ner, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
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

val embeddings_german = WordEmbeddingsModel.pretrained("w2v_cc_300d", "de", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_traffic", "de", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_german, ner, ner_converter))

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.med_ner.traffic").predict("""Put your text here.""")
```

</div>


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_traffic|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|




## Benchmarking
```bash
entity                tp      fp     fn     total  precision  recall  f1
DURATION              113.0   34.0   94.0   207.0     0.7687  0.5459  0.6384
ORGANIZATION_COMPANY  667.0   324.0  515.0  1182.0    0.6731  0.5643  0.6139
LOCATION_CITY         441.0   137.0  166.0  607.0     0.763   0.7265  0.7443
LOCATION_ROUTE        132.0   30.0   61.0   193.0     0.8148  0.6839  0.7437
DATE                  730.0   81.0   168.0  898.0     0.9001  0.8129  0.8543
PERSON                422.0   84.0   174.0  596.0     0.834   0.7081  0.7659
LOCATION_STREET       132.0   12.0   99.0   231.0     0.9167  0.5714  0.704
LOCATION              697.0   94.0   359.0  1056.0    0.8812  0.66    0.7547
TIME                  266.0   34.0   45.0   311.0     0.8867  0.8553  0.8707
TRIGGER               187.0   34.0   192.0  379.0     0.8462  0.4934  0.6233
DISTANCE              99.0    0.0    16.0   115.0     1.0     0.8609  0.9252
NUMBER                608.0   147.0  189.0  797.0     0.8053  0.7629  0.7835
LOCATION_STOP         403.0   53.0   77.0   480.0     0.8838  0.8396  0.8611
macro                   -      -      -       -         -       -     0.6528
micro                   -      -      -       -         -       -     0.7261
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkzOTk3MjM5OF19
-->