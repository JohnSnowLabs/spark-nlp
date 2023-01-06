---
layout: model
title: Spanish Legal Longformer Embeddings (8192 tokens, from mrm8488)
author: John Snow Labs
name: longformer_legal_base_8192
date: 2022-11-26
tags: [longformer, es, spanish, embeddings, transformer, open_source, tensorflow]
task: Embeddings
language: es
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LongformerEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Longformer Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-longformer-base-8192-spanish` is a Spanish model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_legal_base_8192_es_4.2.1_3.0_1669454066237.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
 embeddings = LongformerEmbeddings\
    .pretrained("longformer_legal_base_8192","es")\
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")
```
```scala

val embeddings = LongformerEmbeddings.pretrained("longformer_legal_base_8192","es")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|longformer_legal_base_8192|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|es|
|Size:|575.8 MB|
|Case sensitive:|true|
|Max sentence length:|8192|

## References

https://huggingface.co/mrm8488/legal-longformer-base-8192-spanishT
