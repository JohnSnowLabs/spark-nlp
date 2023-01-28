---
layout: model
title: SDOH Tobacco Usage For Classification
author: John Snow Labs
name: genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli
date: 2023-01-14
tags: [en, licensed, generic_classifier, sdoh, tobacco, clinical]
task: Text Classification
language: en
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
recommended: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Generic Classifier model is intended for detecting tobacco use in clinical notes and trained by using GenericClassifierApproach annotator. `Present:` if the patient was a current consumer of tobacco. `Past:` the patient was a consumer in the past and had quit. `Never:` if the patient had never consumed tobacco. `None: ` if there was no related text.

## Predicted Entities

`Present`, `Past`, `Never`, `None`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli_en_4.2.4_3.0_1673697468673.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli_en_4.2.4_3.0_1673697468673.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
        
sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", 'en','clinical/models')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")

features_asm = FeaturesAssembler()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("features")

generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier    
])

text_list = ["Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. He uses alcohol and cigarettes",
             "The patient quit smoking approximately two years ago with an approximately a 40 pack year history, mostly cigar use. He also reports 'heavy alcohol use', quit 15 months ago.",
             "The patient denies any history of smoking or alcohol abuse. She lives with her one daughter.",
             "She was previously employed as a hairdresser, though says she hasnt worked in 4 years. Not reported by patient, but there is apparently a history of alochol abuse."]

df = spark.createDataFrame(text_list, StringType()).toDF("text")

result = pipeline.fit(df).transform(df)

result.select("text", "class.result").show(truncate=100)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
        
val sentence_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")

val features_asm = new FeaturesAssembler()
    .setInputCols("sentence_embeddings")
    .setOutputCol("features")

val generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli", "en", "clinical/models")
    .setInputCols("features")
    .setOutputCol("class")

val pipeline = new PipelineModel().setStages(Array(
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier))

val data = Seq("Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. He uses alcohol and cigarettes.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash

+----------------------------------------------------------------------------------------------------+---------+
|                                                                                                text|   result|
+----------------------------------------------------------------------------------------------------+---------+
|Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 2...|[Present]|
|The patient quit smoking approximately two years ago with an approximately a 40 pack year history...|   [Past]|
|        The patient denies any history of smoking or alcohol abuse. She lives with her one daughter.|  [Never]|
|She was previously employed as a hairdresser, though says she hasnt worked in 4 years. Not report...|   [None]|
+----------------------------------------------------------------------------------------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[features]|
|Output Labels:|[prediction]|
|Language:|en|
|Size:|3.4 MB|

## Benchmarking

```bash

       label  precision    recall  f1-score   support
       Never       0.89      0.90      0.90       487
        None       0.86      0.78      0.82       269
        Past       0.87      0.79      0.83       415
     Present       0.63      0.82      0.71       203
    accuracy        -         -        0.83      1374
   macro-avg       0.81      0.82      0.81      1374
weighted-avg       0.84      0.83      0.83      1374

```
