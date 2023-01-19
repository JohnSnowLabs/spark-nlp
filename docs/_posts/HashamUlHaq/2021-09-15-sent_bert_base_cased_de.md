---
layout: model
title: BERT Sentence Embeddings German (Base Cased)
author: John Snow Labs
name: sent_bert_base_cased
date: 2021-09-15
tags: [open_source, bert_sentence_embeddings, de]
task: Embeddings
language: de
edition: Spark NLP 3.2.2
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

BERT model trained in German language on a 16GB dataset comprising of Wikipedia dump, EU Bookshop corpus, Open Subtitles, CommonCrawl, ParaCrawl and News Crawl in an MLM fashion.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_base_cased_de_3.2.2_3.0_1631706255661.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_base_cased_de_3.2.2_3.0_1631706255661.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de") \
.setInputCols("sentence") \
.setOutputCol("bert_sentence")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, sent_embeddings ])
```
```scala
val sent_embeddings = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")
.setInputCols("sentence")
.setOutputCol("bert_sentence")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, sent_embeddings ))
```


{:.nlu-block}
```python
import nlu
nlu.load("de.embed_sentence.bert.base_cased").predict("""Put your text here.""")
```

</div>

## Results

```bash
768 dimensional embedding vector per sentence.
```

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
|Language:|de|
|Case sensitive:|true|

## Data Source

This model is imported from https://huggingface.co/dbmdz/bert-base-german-cased