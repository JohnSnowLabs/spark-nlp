---
layout: model
title: Spanish Legal Longformer Embeddings (from Narrativa)
author: John Snow Labs
name: longformer_legal_embeddings
date: 2022-11-15
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

Pretrained Longformer Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `longformer_legal_embeddings` is a Spanish model originally trained by `Narrative`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/longformer_legal_embeddings_es_4.2.0_3.0_1668492586700.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/longformer_legal_embeddings_es_4.2.0_3.0_1668492586700.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

 embeddings = LongformerEmbeddings\
.pretrained("longformer_legal_embeddings","es")\
.setInputCols(["document", "token"])\
.setOutputCol("embeddings")

```
```scala

val embeddings = LongformerEmbeddings.pretrained("longformer_legal_embeddings","es")
.setInputCols("document", "token") 
.setOutputCol("embeddings")

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|longformer_legal_embeddings|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|es|
|Size:|564.1 MB|
|Case sensitive:|true|
|Max sentence length:|4096|

## References

https://huggingface.co/Narrativa/legal-longformer-base-4096-spanish
