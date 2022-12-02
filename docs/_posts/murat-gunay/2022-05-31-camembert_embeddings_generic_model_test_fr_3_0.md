---
layout: model
title: French CamemBert Embeddings (from osanseviero)
author: John Snow Labs
name: camembert_embeddings_generic_model_test
date: 2022-05-31
tags: [fr, open_source, camembert, embeddings]
task: Embeddings
language: fr
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: CamemBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBert Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `dummy-model-test` is a French model orginally trained by `osanseviero`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_embeddings_generic_model_test_fr_3.4.4_3.0_1653991144897.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = CamemBertEmbeddings.pretrained("camembert_embeddings_generic_model_test","fr") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["J'adore Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = CamemBertEmbeddings.pretrained("camembert_embeddings_generic_model_test","fr") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("J'adore Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_embeddings_generic_model_test|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|fr|
|Size:|266.8 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/osanseviero/dummy-model-test