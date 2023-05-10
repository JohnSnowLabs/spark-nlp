---
layout: model
title: Hungarian Legal Roberta Embeddings
author: John Snow Labs
name: roberta_base_hungarian_legal
date: 2023-02-16
tags: [hu, hungarian, embeddings, transformer, open_source, legal, tensorflow]
task: Embeddings
language: hu
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

Pretrained Legal Roberta Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-hungarian-roberta-base` is a Hungarian model originally trained by `joelito`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_hungarian_legal_hu_4.2.4_3.0_1676558480899.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_hungarian_legal_hu_4.2.4_3.0_1676558480899.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_hungarian_legal", "hu")\
  .setInputCols(["sentence"])\
  .setOutputCol("embeddings")
```
```scala
val sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_base_hungarian_legal", "hu")
  .setInputCols("sentence")
  .setOutputCol("embeddings"))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_hungarian_legal|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|hu|
|Size:|416.0 MB|
|Case sensitive:|true|

## References

https://huggingface.co/joelito/legal-hungarian-roberta-base