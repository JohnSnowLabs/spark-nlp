---
layout: model
title: Detect Assertion Status from Response to Treatment
author: John Snow Labs
name: assertion_oncology_response_to_treatment_wip
date: 2022-10-01
tags: [licensed, clinical, oncology, en, assertion]
task: Assertion Status
language: en
edition: Healthcare NLP 4.1.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model detects the assertion status of entities related to response to treatment. The model identifies positive mentions (Present_Or_Past status), and hypothetical or absent mentions (Hypothetical_Or_Absent status).

## Predicted Entities

`Hypothetical_Or_Absent`, `Present_Or_Past`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/assertion_oncology_response_to_treatment_wip_en_4.1.0_3.0_1664641698152.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    .setOutputCol("ner_chunk")    .setWhiteList(["Response_To_Treatment"])
    
assertion = AssertionDLModel.pretrained("assertion_oncology_response_to_treatment_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")
        
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter,
                            assertion])

data = spark.createDataFrame([["The patient presented no evidence of recurrence."]]).toDF("text")

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
    .setWhiteList(Array("Response_To_Treatment"))

val clinical_assertion = AssertionDLModel.pretrained("assertion_oncology_response_to_treatment_wip","en","clinical/models")
    .setInputCols(Array("sentence","ner_chunk","embeddings"))
    .setOutputCol("assertion")
        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                                              sentence_detector,
                                              tokenizer,
                                              word_embeddings,
                                              ner,
                                              ner_converter,
                                              assertion))

val data = Seq("""The patient presented no evidence of recurrence.""").toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
| chunk      | ner_label             | assertion              |
|:-----------|:----------------------|:-----------------------|
| recurrence | Response_To_Treatment | Hypothetical_Or_Absent |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|assertion_oncology_response_to_treatment_wip|
|Compatibility:|Healthcare NLP 4.1.0+|
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
Hypothetical_Or_Absent       0.83    0.96      0.89     46.0
       Present_Or_Past       0.94    0.79      0.86     43.0
             macro-avg       0.89    0.87      0.87     89.0
          weighted-avg       0.89    0.88      0.88     89.0
```
