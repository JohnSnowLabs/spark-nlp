---
layout: model
title: BERT Large Cased
author: John Snow Labs
name: bert_large_cased
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_large_cased_en_2.4.0_2.4_1580580251298.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

embeddings = BertEmbeddings.pretrained("bert_large_cased", "en") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```

```scala

val embeddings = BertEmbeddings.pretrained("bert_large_cased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_large_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|1024|
|Case sensitive:|true|


{:.h2_title}
## Data Source
The model is imported from [https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1](https://tfhub.dev/google/bert_cased_L-24_H-1024_A-16/1)
