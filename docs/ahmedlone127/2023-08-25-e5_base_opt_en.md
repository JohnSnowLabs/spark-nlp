---
layout: model
title: E5 Base Sentence Embeddings Optimized
author: John Snow Labs
name: e5_base_opt
date: 2023-08-25
tags: [en, open_source, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.1.0
spark_version: 3.0
supported: true
engine: onnx
annotator: E5Embeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Text Embeddings by Weakly-Supervised Contrastive Pre-training. Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, Furu Wei, arXiv 2022

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/e5_base_opt_en_5.1.0_3.0_1692963694288.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/e5_base_opt_en_5.1.0_3.0_1692963694288.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings =E5Embeddings.pretrained("e5_base_opt","en") \
            .setInputCols(["documents"]) \
            .setOutputCol("instructor")

pipeline = Pipeline().setStages([document_assembler, embeddings])
```
```scala
val embeddings = E5Embeddings.pretrained("e5_base_opt","en")
      .setInputCols(["document"])
      .setOutputCol("e5_embeddings")
val pipeline = new Pipeline().setStages(Array(document, embeddings))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|e5_base_opt|
|Compatibility:|Spark NLP 5.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[e5]|
|Language:|en|
|Size:|258.7 MB|