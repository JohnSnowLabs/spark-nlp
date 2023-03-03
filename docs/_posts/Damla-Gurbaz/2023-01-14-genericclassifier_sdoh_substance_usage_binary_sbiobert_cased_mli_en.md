---
layout: model
title: SDOH Substance Usage For Binary Classification
author: John Snow Labs
name: genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli
date: 2023-01-14
tags: [en, licensed, generic_classifier, sdoh, substance, clinical]
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

This Generic Classifier model is intended for detecting substance use in clinical notes and trained by using GenericClassifierApproach annotator. `Present:` if the patient was a current consumer of substance or the patient was a consumer in the past and had quit or if the patient had never consumed substance. `None:` if there was no related text.

## Predicted Entities

`Present`, `None`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli_en_4.2.4_3.0_1673697973649.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli_en_4.2.4_3.0_1673697973649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier    
])

text_list = ["Lives in apartment with 16-year-old daughter. Denies EtOH use currently although reports occasional use in past. Utox on admission positive for opiate (on as rx) as well as cocaine. 4-6 cigarettes a day on and off for 10 years. Denies h/o illicit drug use besides marijuana although admitted to cocaine use after being found to have urine positive for cocaine.",
             "The patient quit smoking approximately two years ago with an approximately a 40 pack year history, mostly cigar use. He also reports 'heavy alcohol use', quit 15 months ago."]
             
            
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

val generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli", "en", "clinical/models")
    .setInputCols("features")
    .setOutputCol("class")

val pipeline = new PipelineModel().setStages(Array(
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier))

val data = Seq("The patient quit smoking approximately two years ago with an approximately a 40 pack year history, mostly cigar use. He also reports 'heavy alcohol use', quit 15 months ago.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash

+----------------------------------------------------------------------------------------------------+---------+
|                                                                                                text|   result|
+----------------------------------------------------------------------------------------------------+---------+
|Lives in apartment with 16-year-old daughter. Denies EtOH use currently although reports occasion...|[Present]|
|The patient quit smoking approximately two years ago with an approximately a 40 pack year history...|   [None]|
+----------------------------------------------------------------------------------------------------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli|
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
        None       0.91      0.83      0.87       898
     Present       0.76      0.87      0.81       540
    accuracy        -         -        0.85      1438
   macro-avg       0.83      0.85      0.84      1438
weighted-avg       0.85      0.85      0.85      1438

```