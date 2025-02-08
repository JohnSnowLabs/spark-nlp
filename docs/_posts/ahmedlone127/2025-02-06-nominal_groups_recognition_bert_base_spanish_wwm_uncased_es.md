---
layout: model
title: Castilian, Spanish nominal_groups_recognition_bert_base_spanish_wwm_uncased BertForTokenClassification from ALazcanoG
author: John Snow Labs
name: nominal_groups_recognition_bert_base_spanish_wwm_uncased
date: 2025-02-06
tags: [es, open_source, onnx, token_classification, bert, ner]
task: Named Entity Recognition
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nominal_groups_recognition_bert_base_spanish_wwm_uncased` is a Castilian, Spanish model originally trained by ALazcanoG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nominal_groups_recognition_bert_base_spanish_wwm_uncased_es_5.5.1_3.0_1738828158635.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nominal_groups_recognition_bert_base_spanish_wwm_uncased_es_5.5.1_3.0_1738828158635.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

tokenClassifier  = BertForTokenClassification.pretrained("nominal_groups_recognition_bert_base_spanish_wwm_uncased","es") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("ner")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, tokenClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("nominal_groups_recognition_bert_base_spanish_wwm_uncased", "es")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("ner") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nominal_groups_recognition_bert_base_spanish_wwm_uncased|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|409.7 MB|

## References

https://huggingface.co/ALazcanoG/nominal-groups-recognition-bert-base-spanish-wwm-uncased