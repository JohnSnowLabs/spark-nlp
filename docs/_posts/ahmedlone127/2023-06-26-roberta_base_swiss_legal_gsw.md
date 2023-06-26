---
layout: model
title: Swiss Legal Roberta Embeddings
author: John Snow Labs
name: roberta_base_swiss_legal
date: 2023-06-26
tags: [gsw, swiss, embeddings, transformer, open_source, legal, onnx]
task: Embeddings
language: gsw
edition: Spark NLP 5.0.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Legal Roberta Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-swiss-roberta-base` is a Swiss model originally trained by `joelito`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_swiss_legal_gsw_5.0.0_3.0_1687788882271.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_swiss_legal_gsw_5.0.0_3.0_1687788882271.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_swiss_legal", "gsw")\
  .setInputCols(["sentence"])\
  .setOutputCol("embeddings")
```
```scala
val sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_swiss_legal", "gsw")
  .setInputCols("sentence")
  .setOutputCol("embeddings"))
```
</div>

{:.model-param}

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_swiss_legal", "gsw")\
  .setInputCols(["sentence"])\
  .setOutputCol("embeddings")
```
```scala
val sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_swiss_legal", "gsw")
  .setInputCols("sentence")
  .setOutputCol("embeddings"))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_swiss_legal|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|gsw|
|Size:|692.1 MB|
|Case sensitive:|true|