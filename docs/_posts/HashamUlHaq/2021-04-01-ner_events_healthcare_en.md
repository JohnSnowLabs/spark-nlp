---
layout: model
title: Detect clinical events
author: John Snow Labs
name: ner_events_healthcare
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


Detect clinical events like Date, Occurance, Clinical_Department and a lot more using pretrained NER model.


## Predicted Entities


`OCCURRENCE`, `TREATMENT`, `TIME`, `DATE`, `PROBLEM`, `CLINICAL_DEPT`, `DURATION`, `EVIDENTIAL`, `FREQUENCY`, `TEST`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_healthcare_en_3.0.0_3.0_1617260839291.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_events_healthcare_en_3.0.0_3.0_1617260839291.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_events_healthcare", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
    
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])

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

val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare_100d", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_events_healthcare", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
 	.setInputCols(Array("sentence", "token", "ner"))
 	.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.events_healthcre").predict("""Put your text here.""")
```

</div>


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_events_healthcare|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|




## Benchmarking
```bash
entity      tp     fp     fn   total precision  recall      f1
DURATION   575.0  263.0  231.0   806.0    0.6862  0.7134  0.6995
PROBLEM  8067.0 2479.0 2305.0 10372.0    0.7649  0.7778  0.7713
DATE  1787.0  508.0  315.0  2102.0    0.7786  0.8501  0.8128
CLINICAL_DEPT  1804.0  393.0  338.0  2142.0    0.8211  0.8422  0.8315
OCCURRENCE  1917.0  893.0 2188.0  4105.0    0.6822   0.467  0.5544
TREATMENT  4578.0 1596.0 1817.0  6395.0    0.7415  0.7159  0.7285
FREQUENCY   145.0   46.0  213.0   358.0    0.7592   0.405  0.5282
TEST  3723.0  949.0 1113.0  4836.0    0.7969  0.7699  0.7831
EVIDENTIAL   334.0   80.0  279.0   613.0    0.8068  0.5449  0.6504
macro     -      -      -       -        -       -     0.60759
micro     -      -      -       -        -       -     0.73065
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQ0OTY4MDg1MV19
-->