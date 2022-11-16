---
layout: model
title: Spanish Electra Legal Word Embeddings Small model
author: John Snow Labs
name: legalectra_small
date: 2022-07-08
tags: [open_source, legalectra, embeddings, electra, legal, small, es]
task: Embeddings
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Spanish Legal Word Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legalectra-small-spanish` is a English model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/legalectra_small_es_4.0.0_3.0_1657294835823.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")
  
electra = BertEmbeddings.pretrained("legalectra_small","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, electra])

data = spark.createDataFrame([["Amo a Spark NLP."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val electra = BertEmbeddings.pretrained("legalectra_small","es") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, electra))

val data = Seq("Amo a Spark NLP.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legalectra_small|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|es|
|Size:|51.8 MB|
|Case sensitive:|true|

## References

https://huggingface.co/mrm8488/legalectra-small-spanish
