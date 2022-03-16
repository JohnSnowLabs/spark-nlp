---
layout: model
title: Dutch BERT Sentence Base Cased Embedding
author: John Snow Labs
name: sent_bert_base_cased
date: 2021-09-06
tags: [dutch, open_source, bert_sentence_embeddings, cased, nl]
task: Embeddings
language: nl
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BERTje is a Dutch pre-trained BERT model developed at the University of Groningen.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_cased_nl_3.2.2_3.0_1630926264607.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "nl") \
      .setInputCols("sentence") \
      .setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "nl")
      .setInputCols("sentence")
      .setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_base_cased|
|Compatibility:|Spark NLP 3.2.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[bert_sentence]|
|Language:|nl|
|Case sensitive:|true|

## Data Source

For pre-training, we combined several corpora of high quality Dutch text, listed below. The sizes in
parentheses are the uncompressed text sizes after cleaning.
- Books: a collection of contemporary and historical fiction novels (4.4GB)
- TwNC (Ordelman et al., 2007): a Multifaceted Dutch News Corpus (2.4GB)
- SoNaR-500 (Oostdijk et al., 2013): a multi-genre reference corpus (2.2GB)
- Web news: all articles of 4 Dutch news websites from January 1, 2015 to October 1, 2019 (1.6GB)
- Wikipedia: the October 2019 dump (1.5GB)
