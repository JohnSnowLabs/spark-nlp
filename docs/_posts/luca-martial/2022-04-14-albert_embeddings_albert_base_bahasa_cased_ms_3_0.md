---
layout: model
title: Malay ALBERT Embeddings (Base)
author: John Snow Labs
name: albert_embeddings_albert_base_bahasa_cased
date: 2022-04-14
tags: [albert, embeddings, ms, open_source]
task: Embeddings
language: ms
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: AlBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ALBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `albert-base-bahasa-cased` is a Malay model orginally trained by `malay-huggingface`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_embeddings_albert_base_bahasa_cased_ms_3.4.2_3.0_1649954337875.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = AlbertEmbeddings.pretrained("albert_embeddings_albert_base_bahasa_cased","ms") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Saya suka Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = AlbertEmbeddings.pretrained("albert_embeddings_albert_base_bahasa_cased","ms") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Saya suka Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ms.embed.albert_base_bahasa_cased").predict("""Saya suka Spark NLP""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_embeddings_albert_base_bahasa_cased|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ms|
|Size:|45.7 MB|
|Case sensitive:|false|

## References

- https://huggingface.co/malay-huggingface/albert-base-bahasa-cased
- https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean
- https://github.com/huseinzol05/malay-dataset/tree/master/corpus/pile
- https://github.com/huseinzol05/Malaya/tree/master/pretrained-model/albert