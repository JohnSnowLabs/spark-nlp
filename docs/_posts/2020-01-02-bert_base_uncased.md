---
layout: model
title: BERT Base Uncased
author: John Snow Labs
name: bert_base_uncased
date: 2020-01-02
tags: [open_source, embeddings, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model contains a deep bidirectional transformer trained on Wikipedia and the BookCorpus. The details are described in the paper "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)".

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_cased_en_2.4.0_2.4_1580579557778.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

embeddings = BertEmbeddings.pretrained("bert_base_uncased", "en") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```

```scala

val embeddings = BertEmbeddings.pretrained("bert_base_uncased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_uncased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|false|

{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1](https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1)
