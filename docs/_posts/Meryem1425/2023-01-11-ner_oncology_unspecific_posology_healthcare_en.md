---
layout: model
title: Extract Cancer Therapies and Posology Information
author: John Snow Labs
name: ner_oncology_unspecific_posology_healthcare
date: 2023-01-11
tags: [licensed, clinical, oncology, en, ner, treatment, posology]
task: Named Entity Recognition
language: en
nav_key: models
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts mentions of treatments and posology information using unspecific labels (low granularity).

## Predicted Entities

`Posology_Information`, `Cancer_Therapy`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_unspecific_posology_healthcare_en_4.2.4_3.0_1673475870938.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_unspecific_posology_healthcare_en_4.2.4_3.0_1673475870938.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel\
    .pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel()\
    .pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel\
    .pretrained("ner_oncology_unspecific_posology_healthcare", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
        
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter])

data = spark.createDataFrame([["The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses. She is currently receiving his second cycle of chemotherapy and is in good overall condition."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel
    .pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel()
    .pretrained("embeddings_healthcare_100d", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_unspecific_posology_healthcare", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")
        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                    sentence_detector,
                    tokenizer,
                    word_embeddings,
                    ner,
                    ner_converter))    

val data = Seq("The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses. She is currently receiving his second cycle of chemotherapy and is in good overall condition.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk            | ner_label            |
|:-----------------|:---------------------|
| adriamycin       | Cancer_Therapy       |
| 60 mg/m2         | Posology_Information |
| cyclophosphamide | Cancer_Therapy       |
| 600 mg/m2        | Posology_Information |
| over six courses | Posology_Information |
| second cycle     | Posology_Information |
| chemotherapy     | Cancer_Therapy       |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_unspecific_posology_healthcare|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|33.8 MB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
               label   tp  fp  fn  total  precision  recall   f1
Posology_Information 1435 102 210   1645       0.93    0.87 0.90
      Cancer_Therapy 1281 116 125   1406       0.92    0.91 0.91
           macro-avg 2716 218 335   3051       0.93    0.89 0.91
           micro-avg 2716 218 335   3051       0.93    0.89 0.91
```
