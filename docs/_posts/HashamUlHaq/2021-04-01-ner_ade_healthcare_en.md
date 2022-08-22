---
layout: model
title: Detect Adverse Drug Events (healthcare)
author: John Snow Labs
name: ner_ade_healthcare
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect adverse drug events in tweets, reviews, and medical text using pretrained NER model.

## Predicted Entities

`DRUG`, `ADE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_ade_healthcare_en_3.0.0_3.0_1617260836627.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_healthcare", "en", "clinical/models")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_ade_healthcare", "en", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala

...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_ade_healthcare", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val result = pipeline.fit(Seq.empty[String]).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.med_ner.ade.ade_healthcare").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_ade_healthcare|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|


## Benchmarking
```bash
+------+------+------+------+-------+---------+------+------+
|entity|    tp|    fp|    fn|  total|precision|recall|    f1|
+------+------+------+------+-------+---------+------+------+
|  DRUG|9649.0| 884.0|9772.0|19421.0|   0.9161|0.4968|0.6443|
|   ADE|5909.0|9508.0|1987.0| 7896.0|   0.3833|0.7484|0.5069|
+------+------+------+------+-------+---------+------+------+

+------------------+
|             macro|
+------------------+
|0.5755909944827655|
+------------------+

+------------------+
|             micro|
+------------------+
|0.6045600310939989|
+------------------+
```