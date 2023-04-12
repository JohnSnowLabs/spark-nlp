---
layout: model
title: English Legal BERT Embeddings 
author: John Snow Labs
name: bert_embeddings_legalbert_adept
date: 2023-02-21
tags: [bert, en, english, embeddings, transformer, open_source]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BERT Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legalbert-adept` is a English model originally trained by `hatemestinbejaia`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_embeddings_legalbert_adept_en_4.3.0_3.0_1676979162440.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_embeddings_legalbert_adept_en_4.3.0_3.0_1676979162440.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 documentAssembler = nlp.DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

tokenizer = nlp.Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")
  
embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_legalbert_adept","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = nlp.Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("bert_embeddings_legalbert_adept","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_embeddings_legalbert_adept|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|en|
|Size:|410.1 MB|
|Case sensitive:|true|

## References

https://huggingface.co/hatemestinbejaia/legalbert-adept
