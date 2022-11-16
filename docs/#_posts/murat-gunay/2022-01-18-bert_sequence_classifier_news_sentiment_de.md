---
layout: model
title: Sentiment Analysis of German news
author: John Snow Labs
name: bert_sequence_classifier_news_sentiment
date: 2022-01-18
tags: [german, sentiment, bert_sequence, de, open_source]
task: Sentiment Analysis
language: de
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/mdraw/german-news-sentiment-bert)) and it's been finetuned on news texts about migration for German language, leveraging `Bert` embeddings and `BertForSequenceClassification` for text classification purposes.

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_news_sentiment_de_3.3.4_3.0_1642504435983.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_sequence_classifier_news_sentiment', 'de') \
.setInputCols(['token', 'document']) \
.setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['Die Zahl der Flüchtlinge in Deutschland steigt von Tag zu Tag.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_news_sentiment", "de")
.setInputCols("document", "token")
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["Die Zahl der Flüchtlinge in Deutschland steigt von Tag zu Tag."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.classify.news_sentiment.bert").predict("""Die Zahl der Flüchtlinge in Deutschland steigt von Tag zu Tag.""")
```

</div>

## Results

```bash
['neutral']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_news_sentiment|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|408.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## Data Source

[https://wortschatz.uni-leipzig.de/en/download/German](https://wortschatz.uni-leipzig.de/en/download/German)