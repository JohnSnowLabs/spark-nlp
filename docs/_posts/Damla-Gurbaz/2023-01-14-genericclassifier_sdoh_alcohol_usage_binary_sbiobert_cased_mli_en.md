---
layout: model
title: SDOH Alcohol Usage For Binary Classification
author: John Snow Labs
name: genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli
date: 2023-01-14
tags: [en, licensed, generic_classifier, sdoh, alcohol, clinical]
task: Text Classification
language: en
nav_key: models
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
recommended: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Generic Classifier model is intended for detecting alcohol use in clinical notes and trained by using GenericClassifierApproach annotator. `Present:` if the patient was a current consumer of alcohol or the patient was a consumer in the past and had quit. `Never:` if the patient had never consumed alcohol. `None: ` if there was no related text.

## Predicted Entities

`Present`, `Never`, `None`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli_en_4.2.4_3.0_1673699002618.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli_en_4.2.4_3.0_1673699002618.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier    
])

text_list = ["Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. He uses alcohol and cigarettes",
             "Employee in neuro departmentin at the Center Hospital 18. Widower since 2001. Current smoker since 20 years. No EtOH or illicits.",
             "Patient smoked 4 ppd x 37 years, quitting 22 years ago. He is widowed, lives alone, has three children."]
         
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

val generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli", "en", "clinical/models")
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
|Employee in neuro departmentin at the Center Hospital 18. Widower since 2001. Current smoker sinc...|  [Never]|
|Patient smoked 4 ppd x 37 years, quitting 22 years ago. He is widowed, lives alone, has three chi...|   [None]|
+----------------------------------------------------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli|
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
       Never       0.85      0.86      0.85       523
        None       0.81      0.82      0.81       341
     Present       0.88      0.86      0.87       516
    accuracy        -         -        0.85      1380
   macro-avg       0.85      0.85      0.85      1380
weighted-avg       0.85      0.85      0.85      1380
```
