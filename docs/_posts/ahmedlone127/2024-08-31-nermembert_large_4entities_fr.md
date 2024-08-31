---
layout: model
title: French nermembert_large_4entities CamemBertForTokenClassification from CATIE-AQ
author: John Snow Labs
name: nermembert_large_4entities
date: 2024-08-31
tags: [fr, open_source, onnx, token_classification, camembert, ner]
task: Named Entity Recognition
language: fr
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForTokenClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nermembert_large_4entities` is a French model originally trained by CATIE-AQ.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nermembert_large_4entities_fr_5.4.2_3.0_1725139953725.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nermembert_large_4entities_fr_5.4.2_3.0_1725139953725.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

tokenClassifier  = CamemBertForTokenClassification.pretrained("nermembert_large_4entities","fr") \
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

val tokenClassifier = CamemBertForTokenClassification.pretrained("nermembert_large_4entities", "fr")
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
|Model Name:|nermembert_large_4entities|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|fr|
|Size:|1.3 GB|

## References

https://huggingface.co/CATIE-AQ/NERmembert-large-4entities