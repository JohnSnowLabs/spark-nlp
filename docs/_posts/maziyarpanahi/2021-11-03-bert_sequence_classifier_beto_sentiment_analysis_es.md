---
layout: model
title: BERT Sequence Classification - Spanish Sentiment Analysis (bert_sequence_classifier_beto_sentiment_analysis)
author: John Snow Labs
name: bert_sequence_classifier_beto_sentiment_analysis
date: 2021-11-03
tags: [bert, beto, sentiment, spanish, es, sequence_classification, open_source]
task: Text Classification
language: es
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

# Sentiment Analysis in Spanish

Model trained with TASS 2020 corpus (around ~5k tweets) of several dialects of Spanish. Base model is [BETO](https://github.com/dccuchile/beto), a BERT model trained in Spanish.

Uses `POS`, `NEG`, `NEU` labels.

## Citation

[this paper](https://arxiv.org/abs/2106.09462)

```
@misc{perez2021pysentimiento,
      title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks},
      author={Juan Manuel PÃ©rez and Juan Carlos Giudici and Franco Luque},
      year={2021},
      eprint={2106.09462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Predicted Entities

`NEG`, `NEU`, `POS`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_beto_sentiment_analysis_es_3.3.2_3.0_1635935957872.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
      .pretrained('bert_sequence_classifier_beto_sentiment_analysis', 'es') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

example = spark.createDataFrame([['¡Me siento muy bien!!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_beto_sentiment_analysis", "es")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("¡Me siento muy bien!!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_beto_sentiment_analysis|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|es|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/finiteautomata/beto-sentiment-analysis](https://huggingface.co/finiteautomata/beto-sentiment-analysis)