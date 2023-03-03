---
layout: model
title: SDOH Economics Status For Binary Classification
author: John Snow Labs
name: genericclassifier_sdoh_economics_binary_sbiobert_cased_mli
date: 2023-01-14
tags: [en, licensed, generic_classifier, sdoh, economics, clinical]
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

This model classifies related to social economics status in the clinical documents and trained by using GenericClassifierApproach annotator. `True:` if the patient was currently employed or unemployed. `False:` if there was no related passage.

## Predicted Entities

`True`, `False`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_economics_binary_sbiobert_cased_mli_en_4.2.4_3.0_1673699299086.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/genericclassifier_sdoh_economics_binary_sbiobert_cased_mli_en_4.2.4_3.0_1673699299086.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_economics_binary_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_embeddings,
    features_asm,
    generic_classifier    
])

text_list = ["Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. He uses alcohol and cigarettes",
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

val generic_classifier = GenericClassifierModel.pretrained("genericclassifier_sdoh_economics_binary_sbiobert_cased_mli", "en", "clinical/models")
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
+----------------------------------------------------------------------------------------------------+-------+
|                                                                                                text| result|
+----------------------------------------------------------------------------------------------------+-------+
|Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 2...| [True]|
|The patient quit smoking approximately two years ago with an approximately a 40 pack year history...|[False]|
+----------------------------------------------------------------------------------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|genericclassifier_sdoh_economics_binary_sbiobert_cased_mli|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[features]|
|Output Labels:|[prediction]|
|Language:|en|
|Size:|3.5 MB|

## Benchmarking

```bash
       label  precision    recall  f1-score   support
       False       0.93      0.85      0.89       894
        True       0.79      0.90      0.84       562
    accuracy        -         -        0.87      1456
   macro-avg       0.86      0.87      0.86      1456
weighted-avg       0.87      0.87      0.87      1456
```