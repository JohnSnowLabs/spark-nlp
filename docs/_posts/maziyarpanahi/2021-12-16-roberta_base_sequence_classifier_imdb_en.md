---
layout: model
title: RoBERTa Sequence Classification Base - IMDB (roberta_base_sequence_classifier_imdb)
author: John Snow Labs
name: roberta_base_sequence_classifier_imdb
date: 2021-12-16
tags: [sequence_classification, imdb, en, english, sentiment, open_source, roberta]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

RoBERTa Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

`roberta_base_sequence_classifier_imdb ` is a fine-tuned RoBERTa model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance. 

We used TFRobertaForSequenceClassification to train this model and used RoBertaForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`neg`, `pos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_sequence_classifier_imdb_en_3.4.0_3.0_1639665401969.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = RoBertaForSequenceClassification \
.pretrained('roberta_base_sequence_classifier_imdb', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val sequenceClassifier = RoBertaForSequenceClassification.pretrained("roberta_base_sequence_classifier_imdb", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.roberta.imdb").predict("""Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.""")
```

</div>

## Results

```bash
* +--------------------+
* |result              |
* +--------------------+
* |[neg, neg]          |
* |[pos, pos, pos, pos]|
* +--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_sequence_classifier_imdb|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|465.6 MB|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)

## Benchmarking

```bash
{
	"eval_loss": 0.17360647022724152,
	"eval_accuracy": 0.95464,
	"eval_f1": 0.9546399999999999,
	"eval_precision": 0.95464,
	"eval_recall": 0.95464,
	"eval_runtime": 184.871,
	"eval_samples_per_second": 135.229,
	"eval_steps_per_second": 4.23,
	"epoch": 3.0
}
```