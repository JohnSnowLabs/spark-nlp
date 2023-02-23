---
layout: model
title: Extract Substance Usage Entities from Social Determinants of Health Texts
author: John Snow Labs
name: ner_sdoh_substance_usage_wip
date: 2023-02-23
tags: [licensed, en, sdoh, ner, clinical, social_determinants, public_health, substance_usage]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts substance usage information related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Smoking`, `Substance_Duration`, `Substance_Use`, `Substance_Quantity`, `Substance_Frequency`, `Alcohol`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_substance_usage_wip_en_4.3.1_3.0_1677186927181.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_substance_usage_wip_en_4.3.1_3.0_1677186927181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_substance_usage_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
    ])

sample_texts = ["He does drink occasional alcohol approximately 5 to 6 alcoholic drinks per month.",
             "He continues to smoke one pack of cigarettes daily, as he has for the past 28 years."]


data = spark.createDataFrame(sample_texts, StringType()).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val clinical_embeddings = WordEmbeddingsModel.pretrained('embeddings_clinical', "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_sdoh_substance_usage_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token","embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
))

val data = Seq("He does drink occasional alcohol approximately 5 to 6 alcoholic drinks per month.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------+-----+---+-------------------+
|chunk           |begin|end|ner_label          |
+----------------+-----+---+-------------------+
|drink           |8    |12 |Alcohol            |
|occasional      |14   |23 |Substance_Frequency|
|alcohol         |25   |31 |Alcohol            |
|5 to 6          |47   |52 |Substance_Quantity |
|alcoholic drinks|54   |69 |Alcohol            |
|per month       |71   |79 |Substance_Frequency|
|smoke           |16   |20 |Smoking            |
|one pack        |22   |29 |Substance_Quantity |
|cigarettes      |34   |43 |Smoking            |
|daily           |45   |49 |Substance_Frequency|
|past 28 years   |70   |82 |Substance_Duration |
+----------------+-----+---+-------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_substance_usage_wip|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|3.0 MB|

## Benchmarking

```bash
              label	   tp	 fp	  fn	total	 precision	  recall	      f1
Substance_Frequency	 52.0	2.0	12.0	 64.0	  0.962963	0.812500	0.881356
            Smoking	 77.0	4.0	 2.0	 79.0	  0.950617	0.974684	0.962500
            Alcohol	327.0	8.0	15.0	342.0	  0.976119	0.956140	0.966027
 Substance_Quantity	 74.0	7.0	12.0	 86.0	  0.913580	0.860465	0.886228
 Substance_Duration	 27.0	7.0	14.0	 41.0	  0.794118	0.658537	0.720000
      Substance_Use	204.0	8.0	 6.0	210.0	  0.962264	0.971429	0.966825
```