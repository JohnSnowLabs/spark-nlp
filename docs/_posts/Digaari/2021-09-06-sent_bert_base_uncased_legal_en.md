---
layout: model
title: Legal BERT Sentence Base Uncased Embedding
author: John Snow Labs
name: sent_bert_base_uncased_legal
date: 2021-09-06
tags: [legal, english, open_source, bert_sentence_embeddings, uncased, en]
task: Embeddings
language: en
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
recommended: true
annotator: BertSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

LEGAL-BERT is a family of BERT models for the legal domain, intended to assist legal NLP research, computational law, and legal technology applications. To pre-train the different variations of LEGAL-BERT, we collected 12 GB of diverse English legal text from several fields (e.g., legislation, court cases, contracts) scraped from publicly available resources. Sub-domains variants (CONTRACTS-, EURLEX-, ECHR-) and/or general LEGAL-BERT perform better than using BERT out of the box for domain-specific tasks. A light-weight model (33% the size of BERT-BASE) pre-trained from scratch on legal data with competitive perfomance is also available.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_uncased_legal_en_3.2.2_3.0_1630926286151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased_legal", "en") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_uncased_legal", "en")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed_sentence.bert.base_uncased_legal").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_uncased_legal|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|en|
|Case sensitive:|true|

## Data Source

The model is imported from: https://huggingface.co/nlpaueb/legal-bert-base-uncased