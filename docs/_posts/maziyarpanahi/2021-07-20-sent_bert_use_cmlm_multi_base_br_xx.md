---
layout: model
title: Universal sentence encoder for 100+ languages trained with CMLM (sent_bert_use_cmlm_multi_base_br)
author: John Snow Labs
name: sent_bert_use_cmlm_multi_base_br
date: 2021-07-20
tags: [embeddings, bert, use, open_source, cmlm, xx, multilingual]
task: Embeddings
language: xx
edition: Spark NLP 3.1.3
spark_version: 2.4
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The universal sentence encoder family of models maps the text into high dimensional vectors that capture sentence-level semantics. Our Multilingual-base bitext retrieval model (multilingual-base-br) is trained using a conditional masked language model described in [1]. The model is intended to be used for text classification, text clustering, semantic textural similarity, etc. The model can be fine-tuned for all of these tasks. The base model employs a 12 layer BERT transformer architecture.


The model extends the BERT transformer architecture that is why we use it with BertSentenceEmbeddings.

[1] Ziyi Yang, Yinfei Yang, Daniel Cer, Jax Law, Eric Darve. [Universal Sentence Representations Learning with Conditional Masked Language Model. November 2020](https://openreview.net/forum?id=WDVD4lUCTzU)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_use_cmlm_multi_base_br_xx_3.1.3_2.4_1626783435472.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_use_cmlm_multi_base_br_xx_3.1.3_2.4_1626783435472.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_use_cmlm_multi_base_br", "xx") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
```
```scala
val embeddings = BertSentenceEmbeddings.pretrained("sent_bert_use_cmlm_multi_base_br", "xx")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('xx.embed_sentence.sent_bert_use_cmlm_multi_base_br').predict(text, output_level='sentence')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_use_cmlm_multi_base_br|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert]|
|Language:|xx|
|Case sensitive:|true|

## Data Source

[https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1)

## Benchmarking

```bash
We evaluate this model on XEVAL, translated SentEval sentence representation benchmark. XEVAL will be publicly available soon.

XEVAL	ar	bg	de	....	zh	15 Languages Average
USE-CMLM-Multilingual-Base	80.6	81.2	82.6	....	81.7	81.2
USE-CMLM-Multilingual-Base + BR	82.6	83.0	84.0	....	83.0	82.8

```
