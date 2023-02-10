---
layout: model
title: Detect Symptoms, Treatments and Other Entities in German
author: John Snow Labs
name: ner_healthcare
date: 2021-03-31
tags: [ner, clinical, licensed, de]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to detect symptoms, treatments and other entities in medical text in German language.

## Predicted Entities

`DIAGLAB_PROCEDURE`, `MEDICAL_SPECIFICATION`, `MEDICAL_DEVICE`, `MEASUREMENT`, `BIOLOGICAL_CHEMISTRY`, `BODY_FLUID`, `TIME_INFORMATION`, `LOCAL_SPECIFICATION`, `BIOLOGICAL_PARAMETER`, `PROCESS`, `MEDICATION`, `DOSING`, `DEGREE`, `MEDICAL_CONDITION`, `PERSON`, `TISSUE`, `STATE_OF_HEALTH`, `BODY_PART`, `TREATMENT`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HEALTHCARE_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_3.0.0_3.0_1617208455368.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_de_3.0.0_3.0_1617208455368.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")\
   .setInputCols(["sentence","token"])\
   .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_healthcare", "de", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

clinical_ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("entities")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, clinical_ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist ein hochmalignes bronchogenes Karzinom")
```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","de","clinical/models")
    .setInputCols(Array("sentence","token"))
    .setOutputCol("embeddings")

val ner = MedicalNerModel.pretrained("ner_healthcare_slim", "de", "clinical/models") 
    .setInputCols(Array("sentence", "token", "embeddings")) 
    .setOutputCol("ner")

val clinical_ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("entities")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, clinical_ner_converter))

val data = Seq("Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist ein hochmalignes bronchogenes Karzinom").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-----------------+---------------------+-----+---+
|chunk            |ner_label            |begin|end|
+-----------------+---------------------+-----+---+
|Kleinzellige     |MEASUREMENT          |4    |15 |
|Bronchialkarzinom|MEDICAL_CONDITION    |17   |33 |
|Kleinzelliger    |MEDICAL_SPECIFICATION|36   |48 |
|Lungenkrebs      |MEDICAL_CONDITION    |50   |60 |
|SCLC             |MEDICAL_CONDITION    |63   |66 |
|Karzinom         |MEDICAL_CONDITION    |103  |110|
+-----------------+---------------------+-----+---+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare|
|Compatibility:|Healthcare NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|de|

## Data Source

Trained on 2010 i2b2/VA challenge on concepts, assertions, and relations in clinical text with *w2v_cc_300d*.

## Benchmarking

```bash
|    | label               |     tp |    fp |   fn | precision|    recall|       f1 |
|---:|--------------------:|-------:|------:|-----:|---------:|---------:|---------:|
|  0 | BIOLOGICAL_PARAMETER|    103 |    52 |   57 | 0.6645   | 0.6438   |  0.654   |
|  1 | BODY_FLUID          |    166 |    16 |   24 | 0.9121   | 0.8737   | 0.8925   |
|  2 | PERSON              |    475 |    74 |  142 | 0.8652   | 0.7699   | 0.8148   |
|  3 | DOSING              |     38 |    14 |   31 | 0.7308   | 0.5507   | 0.6281   |
|  4 | DIAGLAB_PROCEDURE   |    236 |    58 |   68 | 0.8027   | 0.7763   | 0.7893   |
|  5 | BODY_PART           |    690 |    72 |   79 | 0.9055   | 0.8973   | 0.9014   |
|  6 | MEDICATION          |    391 |   117 |  167 | 0.7697   | 0.7007   | 0.7336   |
|  7 | STATE_OF_HEALTH     |    321 |    41 |   76 | 0.8867   | 0.8086   | 0.8458   |
|  8 | LOCAL_SPECIFICATION |     57 |    19 |   24 |   0.75   | 0.7037   | 0.7261   |
|  9 | MEASUREMENT         |    574 |   260 |  222 | 0.6882   | 0.7211   | 0.7043   |
| 10 | TREATMENT           |    476 |   131 |  135 | 0.7842   | 0.7791   | 0.7816   |
| 11 | MEDICAL_CONDITION   |   1741 |   442 |  271 | 0.7975   | 0.8653   |   0.83   |
| 12 | TIME_INFORMATION    |    651 |   126 |  161 | 0.8378   | 0.8017   | 0.8194   |
| 13 | BIOLOGICAL_CHEMISTRY|    192 |    55 |   60 | 0.7773   | 0.7619   | 0.7695   |
```