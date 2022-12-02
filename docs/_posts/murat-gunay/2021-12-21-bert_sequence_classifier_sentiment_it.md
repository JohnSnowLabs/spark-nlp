---
layout: model
title: Sentiment Analysis of Italian texts
author: John Snow Labs
name: bert_sequence_classifier_sentiment
date: 2021-12-21
tags: [italian, sentiment, bert, sequence, it, open_source]
task: Sentiment Analysis
language: it
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned for Italian language, leveraging `Bert` embeddings and `BertForSequenceClassification` for text classification purposes.

## Predicted Entities

`negative`, `positive`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_sentiment_it_3.3.4_2.4_1640079386384.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('bert_sequence_classifier_sentiment', 'it') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['Ho mal di testa e mi sento male.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_sentiment", "it")
      .setInputCols("document", "token")
      .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["Ho mal di testa e mi sento male."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("it.classify.sentiment").predict("""Ho mal di testa e mi sento male.""")
```

</div>

## Results

```bash
['negative']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sentiment|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|it|
|Size:|415.5 MB|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/neuraly/bert-base-italian-cased-sentiment](https://huggingface.co/neuraly/bert-base-italian-cased-sentiment)

## Benchmarking

```bash
label      score
accuracy   0.82
```