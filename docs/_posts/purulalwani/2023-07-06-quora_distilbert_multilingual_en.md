---
layout: model
title: Embeddings For Similarity Search
author: purulalwani
name: quora_distilbert_multilingual
date: 2023-07-06
tags: [en, open_source, tensorflow]
task: Embeddings
language: en
edition: Spark NLP 5.0.0
spark_version: 3.2
supported: false
engine: tensorflow
annotator: DistilBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Copy of https://huggingface.co/sentence-transformers/quora-distilbert-multilingual

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/purulalwani/quora_distilbert_multilingual_en_5.0.0_3.2_1688648417016.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/purulalwani/quora_distilbert_multilingual_en_5.0.0_3.2_1688648417016.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
See -> https://huggingface.co/sentence-transformers/quora-distilbert-multilingual
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|quora_distilbert_multilingual|
|Compatibility:|Spark NLP 5.0.0+|
|License:|Open Source|
|Edition:|Community|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|506.5 MB|
|Case sensitive:|false|