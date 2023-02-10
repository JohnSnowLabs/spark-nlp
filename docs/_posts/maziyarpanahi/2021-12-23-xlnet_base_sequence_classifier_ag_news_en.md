---
layout: model
title: XLNet Sequence Classification Base - AG News (xlnet_base_sequence_classifier_ag_news)
author: John Snow Labs
name: xlnet_base_sequence_classifier_ag_news
date: 2021-12-23
tags: [xlnet, ag_news, sequence_classification, en, english, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: XlnetForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

XLNet Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

`xlnet_base_sequence_classifier_ag_news` is a fine-tuned XLNet model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance. 

We used TFXLNetForSequenceClassification to train this model and used XlnetForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`Business`, `Sci/Tech`, `Sports`, `World`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlnet_base_sequence_classifier_ag_news_en_3.4.0_3.0_1640263463723.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlnet_base_sequence_classifier_ag_news_en_3.4.0_3.0_1640263463723.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = XlnetForSequenceClassification \
.pretrained('xlnet_base_sequence_classifier_ag_news', 'en') \
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

val sequenceClassifier = XlnetForSequenceClassification.pretrained("xlnet_base_sequence_classifier_ag_news", "en")
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
nlu.load("en.classify.ag_news.xlnet").predict("""Disney Comics was a comic book publishing company operated by The Walt Disney Company which ran from 1990 to 1993.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlnet_base_sequence_classifier_ag_news|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|440.1 MB|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)

## Benchmarking

```bash
{
	"eval_loss": 0.18038597702980042,
	"eval_accuracy": 0.9494736842105264,
	"eval_f1": 0.9494736842105264,
	"eval_precision": 0.9494736842105264,
	"eval_recall": 0.9494736842105264,
	"eval_runtime": 159.3132,
	"eval_samples_per_second": 47.705,
	"eval_steps_per_second": 1.494,
	"epoch": 3.0
}
```