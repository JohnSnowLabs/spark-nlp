---
layout: model
title: Detect Assertion Status for Radiology
author: John Snow Labs
name: assertion_dl_radiology
date: 2021-03-18
tags: [assertion, en, licensed, clinical, radiology]
task: Assertion Status
language: en
edition: Spark NLP 2.7.4
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_dl_radiology_en_2.7.4_2.4_1616064137627.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  .setOutputCol("ner_chunk")

radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, word_embeddings, radiology_ner, ner_converter, radiology_assertion])

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = nlpPipeline.fit(empty_data)
text = """FINDINGS:
The heart size is normal.  Mediastinal and hilar contours are normal.  The pulmonary vascularity is normal. There is minimal streaky opacity within the left lower lobe, likely reflective of atelectasis. Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion. No right-sided pleural effusion or pneumothorax is definitively seen. There are mildly displaced fractures of the left lateral 8th and likely 9th ribs.
IMPRESSION:
Mildly displaced fractures of the left 8th and likely 9th lateral ribs. Mild left lower lobe atelectasis and probable trace left pleural effusion."""

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

val radiology_assertion = AssertionDLModel.pretrained("assertion_dl_radiology", "en", "clinical/models")
    .setInputCols(Array("sentence", "ner_chunk", "embeddings"))
    .setOutputCol("assertion")

val nlpPipeline = new Pipeline().setStages(Array(documentAssembler,  sentenceDetector, tokenizer, word_embeddings, radiology_ner, ner_converter, radiology_assertion))

val result = pipeline.fit(Seq.empty["FINDINGS:
The heart size is normal.  Mediastinal and hilar contours are normal.  The pulmonary vascularity is normal. There is minimal streaky opacity within the left lower lobe, likely reflective of atelectasis. Blunting of the left costophrenic angle on the lateral view posteriorly suggests a small left pleural effusion. No right-sided pleural effusion or pneumothorax is definitively seen. There are mildly displaced fractures of the left lateral 8th and likely 9th ribs.
IMPRESSION:
Mildly displaced fractures of the left 8th and likely 9th lateral ribs. Mild left lower lobe atelectasis and probable trace left pleural effusion."].toDS.toDF("text")).transform(data)
```
</div>

## Results

```bash
+-----------------------+-----+---+---------------+-------+---------+
|chunk                  |begin|end|ner_label      |sent_id|assertion|
+-----------------------+-----+---+---------------+-------+---------+
|heart size is normal   |14   |33 |ImagingFindings|0      |Confirmed|
|Mediastinal            |37   |47 |BodyPart       |0      |Confirmed|
|hilar                  |53   |57 |BodyPart       |0      |Confirmed|
|contours are normal    |59   |77 |ImagingFindings|0      |Confirmed|
|pulmonary vascularity  |85   |105|BodyPart       |0      |Confirmed|
|normal                 |110  |115|ImagingFindings|0      |Confirmed|
|minimal streaky opacity|127  |149|ImagingFindings|0      |Confirmed|
|left                   |162  |165|Direction      |0      |Confirmed|
|lower lobe             |167  |176|BodyPart       |0      |Confirmed|
|atelectasis            |200  |210|ImagingFindings|0      |Confirmed|
|Blunting               |213  |220|ImagingFindings|1      |Confirmed|
|left                   |229  |232|Direction      |1      |Confirmed|
|costophrenic angle     |234  |251|BodyPart       |1      |Confirmed|
|lateral                |260  |266|Direction      |1      |Confirmed|
|posteriorly            |273  |283|Direction      |1      |Suspected|
|left                   |302  |305|Direction      |1      |Suspected|
|pleural                |307  |313|BodyPart       |1      |Suspected|
|effusion               |315  |322|ImagingFindings|1      |Suspected|
|right-sided            |328  |338|Direction      |2      |Negative |
|pleural                |340  |346|BodyPart       |2      |Negative |
+-----------------------+-----+---+---------------+-------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_dl_radiology|
|Compatibility:|Spark NLP 2.7.4+|
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