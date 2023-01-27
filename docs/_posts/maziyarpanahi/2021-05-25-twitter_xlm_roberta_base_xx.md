---
layout: model
title: Twitter XLM-RoBERTa Base (twitter_xlm_roberta_base)
author: John Snow Labs
name: twitter_xlm_roberta_base
date: 2021-05-25
tags: [xx, multilingual, embeddings, xlm_roberta, open_source]
task: Embeddings
language: xx
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[XLM-RoBERTa](https://ai.facebook.com/blog/-xlm-r-state-of-the-art-cross-lingual-understanding-through-self-supervision/) is a scaled cross-lingual sentence encoder. It is trained on 2.5T of data across 100 languages data filtered from Common Crawl. XLM-R achieves state-of-the-arts results on multiple cross-lingual benchmarks.

The XLM-RoBERTa model was proposed in [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 

It is based on Facebook's RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/twitter_xlm_roberta_base_xx_3.1.0_2.4_1621962368880.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/twitter_xlm_roberta_base_xx_3.1.0_2.4_1621962368880.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
```
```scala
val embeddings = XlmRoBertaEmbeddings.pretrained("twitter_xlm_roberta_base", "xx")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("xx.embed.xlm.twitter").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|twitter_xlm_roberta_base|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|xx|
|Case sensitive:|true|

## Data Source

- [https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base)
- [https://github.com/cardiffnlp/xlm-t](https://github.com/cardiffnlp/xlm-t)
