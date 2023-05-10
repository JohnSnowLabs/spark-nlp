---
layout: model
title: Latvian Legal Roberta Embeddings
author: John Snow Labs
name: roberta_base_latvian_legal
date: 2023-02-16
tags: [lv, latvian, embeddings, transformer, open_source, legal, tensorflow]
task: Embeddings
language: lv
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
annotator: RoBertaEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Legal Roberta Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-latvian-roberta-base` is a Latvian model originally trained by `joelito`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_latvian_legal_lv_4.2.4_3.0_1676555343069.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_latvian_legal_lv_4.2.4_3.0_1676555343069.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_latvian_legal", "lv")\
  .setInputCols(["sentence"])\
  .setOutputCol("embeddings")
```
```scala
val sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_latvian_legal", "lv")
  .setInputCols("sentence")
  .setOutputCol("embeddings"))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_latvian_legal|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|lv|
|Size:|416.0 MB|
|Case sensitive:|true|

## References

https://huggingface.co/joelito/legal-latvian-roberta-base
