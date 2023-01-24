---
layout: model
title: German BERT Base Uncased Model
author: John Snow Labs
name: bert_base_german_uncased
date: 2021-05-20
tags: [open_source, embeddings, german, de, bert]
task: Embeddings
language: de
edition: Spark NLP 3.1.0
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

The source data for the model consists of a recent Wikipedia dump, EU Bookshop corpus, Open Subtitles, CommonCrawl, ParaCrawl and News Crawl. This results in a dataset with a size of 16GB and 2,350,234,427 tokens. The model is trained with an initial sequence length of 512 subwords and was performed for 1.5M steps.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_base_german_uncased_de_3.1.0_2.4_1621504361619.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_base_german_uncased_de_3.1.0_2.4_1621504361619.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
embeddings = BertEmbeddings.pretrained("bert_base_german_uncased", "de") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings])
```
```scala
val embeddings = BertEmbeddings.pretrained("bert_base_german_uncased", "de")
.setInputCols("sentence", "token")
.setOutputCol("embeddings")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("de.embed.bert.uncased").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_base_german_uncased|
|Compatibility:|Spark NLP 3.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[embeddings]|
|Language:|de|
|Case sensitive:|true|

## Data Source

https://huggingface.co/dbmdz/bert-base-german-uncased

## Benchmarking

```bash
For results on downstream tasks like NER or PoS tagging, please refer to
[this repository](https://github.com/stefan-it/fine-tuned-berts-seq).
```