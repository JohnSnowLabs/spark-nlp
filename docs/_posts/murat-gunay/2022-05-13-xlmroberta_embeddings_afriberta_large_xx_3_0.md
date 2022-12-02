---
layout: model
title: Multilingual XLMRoBerta Embeddings (from castorini)
author: John Snow Labs
name: xlmroberta_embeddings_afriberta_large
date: 2022-05-13
tags: [ha, yo, ig, am, so, open_source, xlm_roberta, embeddings, xx, afriberta]
task: Embeddings
language: xx
edition: Spark NLP 3.4.4
spark_version: 3.0
supported: true
annotator: XlmRoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XLMRoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `afriberta_large` is a Multilingual model orginally trained by `castorini`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_embeddings_afriberta_large_xx_3.4.4_3.0_1652439242600.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
  
embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_afriberta_large","xx") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = XlmRoBertaEmbeddings.pretrained("xlmroberta_embeddings_afriberta_large","xx") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_embeddings_afriberta_large|
|Compatibility:|Spark NLP 3.4.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Size:|471.1 MB|
|Case sensitive:|true|

## References

- https://huggingface.co/castorini/afriberta_large
- https://github.com/keleog/afriberta
