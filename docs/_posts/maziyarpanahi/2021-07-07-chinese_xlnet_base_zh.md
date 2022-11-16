---
layout: model
title: Chinese Pre-Trained XLNet (Base)
author: John Snow Labs
name: chinese_xlnet_base
date: 2021-07-07
tags: [open_source, embeddings, xlnet, zh]
task: Embeddings
language: zh
edition: Spark NLP 3.1.2
spark_version: 2.4
supported: true
annotator: XlnetEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking. The details are described in the paper "[​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)"

This model is based on CMU/Google official XLNet: https://github.com/zihangdai/xlnet

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/chinese_xlnet_base_zh_3.1.2_2.4_1625668114420.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = XlnetEmbeddings.pretrained("chinese_xlnet_base", "zh") \
      .setInputCols("sentence", "token") \
      .setOutputCol("embeddings")
```
```scala
val embeddings = XlnetEmbeddings.pretrained("chinese_xlnet_base", "zh")
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
```

{:.nlu-block}
```python
import nlu

text = ["I love NLP"]
embeddings_df = nlu.load('zh.embed.chinese_xlnet_base').predict(text, output_level='token')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|chinese_xlnet_base|
|Compatibility:|Spark NLP 3.1.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|zh|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/hfl/chinese-xlnet-base](https://huggingface.co/hfl/chinese-xlnet-base)