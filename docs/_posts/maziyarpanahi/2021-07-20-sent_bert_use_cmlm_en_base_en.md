---
layout: model
title: Universal sentence encoder for English trained with CMLM (sent_bert_use_cmlm_en_base)
author: John Snow Labs
name: sent_bert_use_cmlm_en_base
date: 2021-07-20
tags: [embeddings, bert, open_source, english, en, cmlm, use]
task: Embeddings
language: en
edition: Spark NLP 3.1.3
spark_version: 2.4
supported: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Universal sentence encoder for English trained with a conditional masked language model. The universal sentence encoder family of models maps the text into high dimensional vectors that capture sentence-level semantics. Our English-base (en-base) model is trained using a conditional masked language model described in [1]. The model is intended to be used for text classification, text clustering, semantic textual similarity, etc. It can also be used used as modularized input for multimodal tasks with text as a feature. The base model employs a 12 layer BERT transformer architecture.


The model extends the BERT transformer architecture that is why we use it with BertSentenceEmbeddings.

[1] Ziyi Yang, Yinfei Yang, Daniel Cer, Jax Law, Eric Darve. [Universal Sentence Representations Learning with Conditional Masked Language Model. November 2020](https://openreview.net/forum?id=WDVD4lUCTzU)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_use_cmlm_en_base_en_3.1.3_2.4_1626782549609.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_use_cmlm_en_base", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
```
```scala
val embeddings = BertSentenceEmbeddings.pretrained("sent_bert_use_cmlm_en_base", "en")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.sent_bert_use_cmlm_en_base').predict(text, output_level='sentence')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_use_cmlm_en_base|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1](https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1)

## Benchmarking

```bash
Training News dataset by using ClassifierDL with 120K training examples:

precision    recall  f1-score   support

Business       0.84      0.90      0.87      1784
Sci/Tech       0.92      0.85      0.89      2053
Sports       0.98      0.96      0.97      1952
World       0.89      0.93      0.91      1811

accuracy                           0.91      7600
macro avg       0.91      0.91      0.91      7600
weighted avg       0.91      0.91      0.91      7600
```