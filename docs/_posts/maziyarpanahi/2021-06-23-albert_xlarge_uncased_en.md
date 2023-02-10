---
layout: model
title: ALBERT Embeddings (XLarge Uncase)
author: John Snow Labs
name: albert_xlarge_uncased
date: 2021-06-23
tags: [open_source, en, english, embeddings, albert, xlarge]
task: Embeddings
language: en
edition: Spark NLP 3.1.1
spark_version: 2.4
supported: true
annotator: AlBertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation. The details are described in the paper "[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.](https://arxiv.org/abs/1909.11942)"

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_xlarge_uncased_en_3.1.1_2.4_1624450838361.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/albert_xlarge_uncased_en_3.1.1_2.4_1624450838361.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = AlbertEmbeddings.pretrained("albert_xlarge_uncased", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = AlbertEmbeddings.pretrained("albert_xlarge_uncased", "en")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('en.embed.albert.xlarge_uncased').predict(text, output_level='token')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|albert_xlarge_uncased|
|Compatibility:|Spark NLP 3.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[https://huggingface.co/albert-xlarge-v2](https://huggingface.co/albert-xlarge-v2)