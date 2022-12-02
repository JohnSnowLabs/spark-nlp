---
layout: model
title: DeBERTa Sequence Classification Base - IMDB (deberta_v3_base_sequence_classifier_imdb)
author: John Snow Labs
name: deberta_v3_base_sequence_classifier_imdb
date: 2022-04-09
tags: [open_source, deberta, v3, sequence_classification, en, english, imdb]
task: Text Classification
language: en
edition: Spark NLP 3.4.3
spark_version: 3.0
supported: true
annotator: DeBertaForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

DeBERTa v3 model with sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.

deberta_v3_base_sequence_classifier_imdb is a fine-tuned DeBERTa model that is ready to be used for Sequence Classification tasks such as sentiment analysis or multi-class text classification and it achieves state-of-the-art performance.

We used TFDebertaV2ForSequenceClassification to train this model and used DeBertaForSequenceClassification annotator in Spark NLP 🚀 for prediction at scale!

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_base_sequence_classifier_imdb_en_3.4.3_3.0_1649511798244.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\ 
.setInputCol("text")\ 
.setOutputCol("document")

tokenizer = Tokenizer()\ 
.setInputCols(['document'])\ 
.setOutputCol('token') 

sequenceClassifier = DeBertaForSequenceClassification.pretrained("deberta_v3_base_sequence_classifier_imdb", "en")\ 
.setInputCols(["document", "token"])\ 
.setOutputCol("class")\ 
.setCaseSensitive(True)\ 
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

val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols("document")
.setOutputCol("token")

val sequenceClassifier = DeBertaForSequenceClassification.pretrained("deberta_v3_base_sequence_classifier_imdb", "en")
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
nlu.load("en.classify.sentiment.imdb.deberta.base").predict("""I really liked that movie!""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_base_sequence_classifier_imdb|
|Compatibility:|Spark NLP 3.4.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|658.8 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

[https://huggingface.co/datasets/imdb](https://huggingface.co/datasets/imdb)
