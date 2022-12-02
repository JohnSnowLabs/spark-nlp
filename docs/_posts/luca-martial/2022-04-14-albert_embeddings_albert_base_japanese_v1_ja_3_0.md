---
layout: model
title: Japanese ALBERT Embeddings (from ken11)
author: John Snow Labs
name: albert_embeddings_albert_base_japanese_v1
date: 2022-04-14
tags: [albert, embeddings, ja, open_source]
task: Embeddings
language: ja
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: AlBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ALBERT Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `albert-base-japanese-v1` is a Japanese model orginally trained by `ken11`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_embeddings_albert_base_japanese_v1_ja_3.4.2_3.0_1649954238475.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

embeddings = AlbertEmbeddings.pretrained("albert_embeddings_albert_base_japanese_v1","ja") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["私はSpark NLPを愛しています"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = AlbertEmbeddings.pretrained("albert_embeddings_albert_base_japanese_v1","ja") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("私はSpark NLPを愛しています").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("ja.embed.albert_base_japanese_v1").predict("""私はSpark NLPを愛しています""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_embeddings_albert_base_japanese_v1|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[bert]|
|Language:|ja|
|Size:|45.6 MB|
|Case sensitive:|false|

## References

- https://huggingface.co/ken11/albert-base-japanese-v1
- https://ken11.jp/blog/sentencepiece-tokenizer-bug
- https://ja.wikipedia.org/wiki/Wikipedia:%E3%83%87%E3%83%BC%E3%82%BF%E3%83%99%E3%83%BC%E3%82%B9%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89
- https://www.rondhuit.com/download.html#ldcc
- https://github.com/google/sentencepiece
- https://opensource.org/licenses/MIT