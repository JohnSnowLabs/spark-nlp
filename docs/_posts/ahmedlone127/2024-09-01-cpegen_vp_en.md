---
layout: model
title: English cpegen_vp DistilBertForTokenClassification from Neurona
author: John Snow Labs
name: cpegen_vp
date: 2024-09-01
tags: [en, open_source, onnx, token_classification, distilbert, ner]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: DistilBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cpegen_vp` is a English model originally trained by Neurona.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cpegen_vp_en_5.4.2_3.0_1725171394137.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cpegen_vp_en_5.4.2_3.0_1725171394137.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier  = DistilBertForTokenClassification.pretrained("cpegen_vp","en") \
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

val tokenClassifier = DistilBertForTokenClassification.pretrained("cpegen_vp", "en")
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
|Model Name:|cpegen_vp|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/Neurona/cpegen_vp