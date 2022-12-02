---
layout: model
title: Italian Electra Embeddings (from dbmdz)
author: John Snow Labs
name: electra_embeddings_electra_base_italian_xxl_cased_generator
date: 2022-05-17
tags: [it, open_source, electra, embeddings]
task: Embeddings
language: it
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Electra Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `electra-base-italian-xxl-cased-generator` is a Italian model orginally trained by `dbmdz`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_embeddings_electra_base_italian_xxl_cased_generator_it_3.4.4_3.0_1652786574536.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = BertEmbeddings.pretrained("electra_embeddings_electra_base_italian_xxl_cased_generator","it") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Adoro Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("electra_embeddings_electra_base_italian_xxl_cased_generator","it") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Adoro Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_embeddings_electra_base_italian_xxl_cased_generator|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|it|
|Size:|128.0 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/dbmdz/electra-base-italian-xxl-cased-generator
- http://opus.nlpl.eu/
- https://traces1.inria.fr/oscar/
- https://github.com/dbmdz/berts/issues/7
- https://github.com/stefan-it/turkish-bert/tree/master/electra
- https://github.com/stefan-it/italian-bertelectra
- https://github.com/dbmdz/berts/issues/new