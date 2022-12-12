---
layout: model
title: Detect Assertion Status from Oncology Entities
author: John Snow Labs
name: assertion_oncology_wip
date: 2022-10-11
tags: [licensed, clinical, oncology, en, assertion]
task: Assertion Status
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model detects the assertion status of entities related to oncology (including diagnoses, therapies and tests).

## Predicted Entities

`Absent`, `Family`, `Hypothetical`, `Past`, `Possible`, `Present`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ASSERTION_ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_oncology_wip_en_4.0.0_3.0_1665519630515.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")\
    .setWhiteList(["Cancer_Dx", "Tumor_Finding", "Cancer_Surgery", "Chemotherapy", "Pathology_Test", "Imaging_Test"])
    
assertion = AssertionDLModel.pretrained("assertion_oncology_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
        
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter,
                            assertion])

data = spark.createDataFrame([["The patient is suspected to have breast cancer. Family history is positive for other cancers. The result of the biopsy was positive."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")
    .setWhiteList(Array("Cancer_Dx", "Tumor_Finding", "Cancer_Surgery", "Chemotherapy", "Pathology_Test", "Imaging_Test"))

val clinical_assertion = AssertionDLModel.pretrained("assertion_oncology_wip","en","clinical/models")
    .setInputCols(Array("sentence","ner_chunk","embeddings"))
    .setOutputCol("assertion")
        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                                              sentence_detector,
                                              tokenizer,
                                              word_embeddings,
                                              ner,
                                              ner_converter,
                                              assertion))

val data = Seq("""The patient is suspected to have lung cancer. Family history is positive for other cancers. The result of the biopsy was positive.""").toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
| chunk         | ner_label      | assertion   |
|:--------------|:---------------|:------------|
| lung cancer   | Cancer_Dx      | Possible    |
| cancers       | Cancer_Dx      | Family      |
| biopsy        | Pathology_Test | Past        |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_oncology_wip|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, chunk, embeddings]|
|Output Labels:|[assertion_pred]|
|Language:|en|
|Size:|1.4 MB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
       label  precision  recall  f1-score  support
      Absent       0.83    0.79      0.81    368.0
      Family       0.80    0.80      0.80     40.0
Hypothetical       0.65    0.57      0.61    229.0
        Past       0.90    0.91      0.91   2124.0
    Possible       0.64    0.61      0.63     85.0
     Present       0.87    0.88      0.88   2121.0
   macro-avg       0.78    0.76      0.77   4967.0
weighted-avg       0.87    0.87      0.87   4967.0
```
