---
layout: model
title: Extract Mentions of Response to Cancer Treatment
author: John Snow Labs
name: ner_oncology_response_to_treatment
date: 2022-10-25
tags: [licensed, clinical, oncology, en, ner, treatment]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts entities related to the patient"s response to the oncology treatment, including clinical response and changes in tumor size.

Definitions of Predicted Entities:

- `Line_Of_Therapy`: Explicit references to the line of therapy of an oncological therapy (e.g. "first-line treatment").
- `Response_To_Treatment`: Terms related to clinical progress of the patient related to cancer treatment, including "recurrence", "bad response" or "improvement".
- `Size_Trend`: Terms related to the changes in the size of the tumor (such as "growth" or "reduced in size").



## Predicted Entities

`Line_Of_Therapy`, `Response_To_Treatment`, `Size_Trend`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ONCOLOGY_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_response_to_treatment_en_4.0.0_3.0_1666722959227.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner = MedicalNerModel.pretrained("ner_oncology_response_to_treatment", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter])

data = spark.createDataFrame([["She completed her first-line therapy, but some months later there was recurrence of the breast cancer. "]]).toDF("text")

result = pipeline.fit(data).transform(data)
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
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_response_to_treatment", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

        
val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter))    

val data = Seq("She completed her first-line therapy, but some months later there was recurrence of the breast cancer. ").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
| chunk      | ner_label             |
|:-----------|:----------------------|
| first-line | Line_Of_Therapy       |
| recurrence | Response_To_Treatment |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_response_to_treatment|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|34.4 MB|
|Dependencies:|embeddings_clinical|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
                label  tp  fp  fn  total  precision  recall   f1
Response_To_Treatment 326 101 157    483       0.76    0.67 0.72
           Size_Trend  43  28  70    113       0.61    0.38 0.47
      Line_Of_Therapy  99  11   7    106       0.90    0.93 0.92
            macro_avg 468 140 234    702       0.76    0.66 0.70
            micro_avg 468 140 234    702       0.76    0.67 0.71
```
