---
layout: model
title: Castilian, Spanish nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner BertForTokenClassification from pineiden
author: John Snow Labs
name: nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner
date: 2023-11-07
tags: [bert, es, open_source, token_classification, onnx]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner` is a Castilian, Spanish model originally trained by pineiden.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner_es_5.2.0_3.0_1699389473204.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner_es_5.2.0_3.0_1699389473204.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")
    
    
tokenClassifier = BertForTokenClassification.pretrained("nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner","es") \
            .setInputCols(["documents","token"]) \
            .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenClassifier])

pipelineModel = pipeline.fit(data)

pipelineDF = pipelineModel.transform(data)

```
```scala


val documentAssembler = new DocumentAssembler()
    .setInputCol("text") 
    .setOutputCol("embeddings")
    
val tokenClassifier = BertForTokenClassification  
    .pretrained("nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner", "es")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenClassifier))

val pipelineModel = pipeline.fit(data)

val pipelineDF = pipelineModel.transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nominal_groups_recognition_medical_disease_competencia2_bert_medical_ner|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|407.2 MB|

## References

https://huggingface.co/pineiden/nominal-groups-recognition-medical-disease-competencia2-bert-medical-ner