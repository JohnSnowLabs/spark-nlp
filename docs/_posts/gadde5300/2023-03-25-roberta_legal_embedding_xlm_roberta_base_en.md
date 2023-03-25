---
layout: model
title: English Legal RoBerta Embeddings (Large)
author: John Snow Labs
name: roberta_legal_embedding_xlm_roberta_base
date: 2023-03-25
tags: [en, english, roberta, embeddings, transformer, open_source, tensorflow]
task: Embeddings
language: en
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBerta Embeddings model is a English Legal embeddings model adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_legal_embedding_xlm_roberta_base_en_4.4.0_3.0_1679742723697.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_legal_embedding_xlm_roberta_base_en_4.4.0_3.0_1679742723697.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_legal_embedding_xlm_roberta_base","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = nlp.Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I Love spark nlp"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_legal_embedding_xlm_roberta_base","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I Love spark nlp").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_legal_embedding_xlm_roberta_base|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|695.1 MB|
|Case sensitive:|true|

## References

https://huggingface.co/joelito/legal-xlm-roberta-base