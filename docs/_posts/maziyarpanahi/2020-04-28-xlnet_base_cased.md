---
layout: model
title: XLNet Embeddings (Base)
author: John Snow Labs
name: xlnet_base_cased
date: 2020-04-28
task: Embeddings
language: en
edition: Spark NLP 2.5.0
tags: [embeddings, en, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking. The details are described in the paper "[​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)"

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_base_cased_en_2.5.0_2.4_1588074114942.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

embeddings = XlnetEmbeddings.pretrained("xlnet_base_cased", "en") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```

```scala

val embeddings = XlnetEmbeddings.pretrained("xlnet_base_cased", "en")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlnet_base_cased|
|Type:|embeddings|
|Compatibility:|Spark NLP 2.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[word_embeddings]|
|Language:|[en]|
|Dimension:|768|
|Case sensitive:|true|

{:.h2_title}
## Data Source
The model is imported from [https://github.com/zihangdai/xlnet](https://github.com/zihangdai/xlnet)
