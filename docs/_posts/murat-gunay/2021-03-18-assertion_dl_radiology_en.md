---
layout: model
title: Detect Assertion Status for Radiology
author: John Snow Labs
name: assertion_dl_radiology
date: 2021-03-18
tags: [assertion, en, licensed, radiology, clinical]
task: Assertion Status
language: en
edition: Spark NLP for Healthcare 2.7.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Assign assertion status to clinical entities extracted by Radiology NER based on their context in the text.

## Predicted Entities

`Confirmed`, `Suspected`, `Negative`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_radiology_en_2.7.4_2.4_1616071311532.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Extract radiology entities using the radiology NER model in the pipeline and assign assertion status for them with `assertion_dl_radiology` pretrained model.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
  .setInputCols(["sentence", "token"])\
  .setOutputCol("embeddings")

radiology_ner = NerDLModel.pretrained("ner_radiology", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

ner_converter = NerConverter() \
  .setInputCols(["sentence", "token", "ner"]) \
  .setOutputCol("ner_chunk")\
  .setWhiteList(["ImagingFindings"])

radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, radiology_ner, ner_converter, radiology_assertion])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)
text = """Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion. No right-sided pleural effusion or pneumothorax is definitively seen. There are mildly displaced fractures of the left lateral 8th and likely 9th ribs."""

result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
...
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")

val radiology_ner = NerDLModel.pretrained("ner_radiology", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")

val ner_converter = NerConverter() 
  .setInputCols(Array("sentence", "token", "ner")) 
  .setOutputCol("ner_chunk")
  .setWhiteList(Array("ImagingFindings"))

val radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val nlpPipeline = new Pipeline().setStages(Array(documentAssembler,  sentenceDetector, tokenizer, word_embeddings, radiology_ner, ner_converter, radiology_assertion))

val data = Seq("Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion. No right-sided pleural effusion or pneumothorax is definitively seen. There are mildly displaced fractures of the left lateral 8th and likely 9th ribs.").toDF("text")
val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------------+-------------------+---------------+-------+---------+
|sentences                                                                                                      |chunk              |ner_label      |sent_id|assertion|
+---------------------------------------------------------------------------------------------------------------+-------------------+---------------+-------+---------+
|Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion.|Blunting           |ImagingFindings|0      |Confirmed|
|Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion.|effusion           |ImagingFindings|0      |Suspected|
|No right-sided pleural effusion or pneumothorax is definitively seen.                                          |effusion           |ImagingFindings|1      |Negative |
|No right-sided pleural effusion or pneumothorax is definitively seen.                                          |pneumothorax       |ImagingFindings|1      |Negative |
|There are mildly displaced fractures of the left lateral 8th and likely 9th ribs.                              |displaced fractures|ImagingFindings|2      |Confirmed|
+---------------------------------------------------------------------------------------------------------------+-------------------+---------------+-------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_radiology|
|Compatibility:|Spark NLP for Healthcare 2.7.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|

## Data Source

Custom internal labeled radiology dataset.

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
Suspected	 629	 155	 159	 0.8022959	 0.7982234	 0.80025446
Negative	 417	 53	 36	 0.88723403	 0.9205298	 0.9035753
Confirmed	 2252	 173	 186	 0.9286598	 0.92370796	 0.92617726
tp: 3298 fp: 381 fn: 381 labels: 3
Macro-average	 prec: 0.87272996, rec: 0.88082033, f1: 0.8767565
Micro-average	 prec: 0.89643925, rec: 0.89643925, f1: 0.89643925
```