---
layout: model
title: ALBERT XXLarge Uncase
author: John Snow Labs
name: albert_xxlarge_uncased
date: 2020-04-28
tags: [embeddings, en, albert]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation. The details are described in the paper "[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.](https://arxiv.org/abs/1909.11942)"

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/albert_xxlarge_uncased_en_2.5.0_2.4_1588073588232.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

{% include programmingLanguageSelectScalaPython.html %}

```python

embeddings = AlbertEmbeddings.pretrained("albert_xxlarge_uncased", "en") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```

```scala

val embeddings = AlbertEmbeddings.pretrained("albert_xxlarge_uncased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|albert_xxlarge_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.5.0|
|Edition:|Official|
|Spark inputs:|[sentence, token]|
|Spark outputs:|[word_embeddings]|
|Language:|[en]|
|Dimension:|1024|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Source
The model is imported from [https://tfhub.dev/google/albert_xlarge/3](https://tfhub.dev/google/albert_xlarge/3)
