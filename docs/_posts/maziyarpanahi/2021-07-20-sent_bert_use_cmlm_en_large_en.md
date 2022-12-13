---
layout: model
title: Universal sentence encoder for English trained with CMLM (sent_bert_use_cmlm_en_large)
author: John Snow Labs
name: sent_bert_use_cmlm_en_large
date: 2021-07-20
tags: [embeddings, bert, use, en, english, cmlm, open_source]
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

Universal sentence encoder for English trained with a conditional masked language model. The universal sentence encoder family of models maps the text into high dimensional vectors that capture sentence-level semantics. Our English-Large (en-large) model is trained using a conditional masked language model described in [1]. The model is intended to be used for text classification, text clustering, semantic textual similarity, etc. It can also be used used as modularized input for multimodal tasks with text as a feature. The large model employs a 24 layer BERT transformer architecture.



The model extends the BERT transformer architecture that is why we use it with BertSentenceEmbeddings.

[1] Ziyi Yang, Yinfei Yang, Daniel Cer, Jax Law, Eric Darve. [Universal Sentence Representations Learning with Conditional Masked Language Model. November 2020](https://openreview.net/forum?id=WDVD4lUCTzU)

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_use_cmlm_en_large_en_3.1.3_2.4_1626783107796.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_use_cmlm_en_large_en_3.1.3_2.4_1626783107796.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertSentenceEmbeddings.pretrained("sent_bert_use_cmlm_en_large", "en") \
.setInputCols("sentence") \
.setOutputCol("sentence_embeddings")
```
```scala
val embeddings = BertSentenceEmbeddings.pretrained("sent_bert_use_cmlm_en_large", "en")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
```

{:.nlu-block}
```python
import nlu

text = ["I hate cancer", "Antibiotics aren't painkiller"]
embeddings_df = nlu.load('en.embed_sentence.sent_bert_use_cmlm_en_large').predict(text, output_level='sentence')
embeddings_df
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_use_cmlm_en_large|
|Compatibility:|Spark NLP 3.1.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert]|
|Language:|en|
|Case sensitive:|false|

## Data Source

[https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1](https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1)

## Benchmarking

```bash
Training News dataset by using ClassifierDL with 120K training examples:

precision    recall  f1-score   support

Business       0.88      0.89      0.88      1880
Sci/Tech       0.91      0.88      0.89      1963
Sports       0.98      0.95      0.97      1961
World       0.89      0.94      0.92      1796

accuracy                           0.92      7600
macro avg       0.92      0.92      0.92      7600
weighted avg       0.92      0.92      0.92      7600


We evaluate this model on SentEval sentence representation benchmark.

SentEval	MR	CR	SUBJ	MPQA	SST	TREC	MRPC	SICK-E	SICK-R	Avg
USE-CMLM-Base	83.6	89.9	96.2	89.3	88.5	91.0	69.7	82.3	83.4	86.0
USE-CMLM-Large	85.6	89.1	96.6	89.3	91.4	92.4	70.0	82.2	84.5	86.8
```