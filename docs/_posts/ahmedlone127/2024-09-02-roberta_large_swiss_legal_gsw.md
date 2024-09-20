---
layout: model
title: Swiss Legal Roberta Embeddings
author: John Snow Labs
name: roberta_large_swiss_legal
date: 2024-09-02
tags: [gsw, swiss, embeddings, transformer, open_source, legal, onnx]
task: Embeddings
language: gsw
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Legal Roberta Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `legal-swiss-roberta-large` is a Swiss model originally trained by `joelito`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_large_swiss_legal_gsw_5.5.0_3.0_1725264232623.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_large_swiss_legal_gsw_5.5.0_3.0_1725264232623.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_large_swiss_legal", "gsw")\
  .setInputCols(["sentence"])\
  .setOutputCol("embeddings")
```
```scala
val sentence_embeddings = RoBertaEmbeddings.pretrained("roberta_large_swiss_legal", "gsw")
  .setInputCols("sentence")
  .setOutputCol("embeddings"))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_large_swiss_legal|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[roberta]|
|Language:|gsw|
|Size:|1.6 GB|

## References

References

https://huggingface.co/joelito/legal-swiss-roberta-large