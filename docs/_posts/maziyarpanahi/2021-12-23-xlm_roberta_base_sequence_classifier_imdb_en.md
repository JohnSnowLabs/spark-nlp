---
layout: model
title: XLM-RoBERTa Sequence Classification Base - IMDB (xlm_roberta_base_sequence_classifier_imdb)
author: John Snow Labs
name: xlm_roberta_base_sequence_classifier_imdb
date: 2021-12-23
tags: [sequence_classification, imdb, sentiment, open_source, en, english, xlm_roberta]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

XLM-RoBERTa Model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

`xlm_roberta_base_sequence_classifier_imdb ` is a fine-tuned XLM-RoBERTa model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance. 

We used TFXLMRobertaForSequenceClassification to train this model and used XlmRoBertaForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

## Predicted Entities

`neg`, `pos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_sequence_classifier_imdb_en_3.4.0_3.0_1640258683543.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_sequence_classifier_imdb_en_3.4.0_3.0_1640258683543.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = XlmRoBertaForSequenceClassification \
.pretrained('xlm_roberta_base_sequence_classifier_imdb', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler,
tokenizer,
sequenceClassifier
])

example = spark.createDataFrame([['I really liked that movie!']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val tokenClassifier = XlmRoBertaForSequenceClassification.pretrained("xlm_roberta_base_sequence_classifier_imdb", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I really liked that movie!").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.xlm_roberta.imdb").predict("""I really liked that movie!""")
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
|Model Name:|xlm_roberta_base_sequence_classifier_imdb|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|870.5 MB|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)

## Benchmarking

```bash
{
	"eval_loss": 0.21888355910778046,
	"eval_accuracy": 0.94092,
	"eval_f1": 0.94092,
	"eval_precision": 0.94092,
	"eval_recall": 0.94092,
	"eval_runtime": 323.4878,
	"eval_samples_per_second": 77.283,
	"eval_steps_per_second": 4.832,
	"epoch": 2.0
}
```