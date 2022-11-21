---
layout: model
title: DistilBERT Sequence Classification - Emotion (distilbert_sequence_classifier_emotion)
author: John Snow Labs
name: distilbert_sequence_classifier_emotion
date: 2021-11-21
tags: [distilbert, emotion, en, english, sequence_classification, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

[Distilbert](https://arxiv.org/abs/1910.01108) is created with knowledge distillation during the pre-training phase which reduces the size of a BERT model by 40% while retaining 97% of its language understanding. It's smaller, faster than Bert and any other Bert-based model.

## Predicted Entities

`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_emotion_en_3.3.3_3.0_1637500005725.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = DistilBertForSequenceClassification \
.pretrained('distilbert_sequence_classifier_emotion', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

example = spark.createDataFrame([['I like you.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_emotion", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I like you.").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distilbert_sequence.emotion").predict("""I like you.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_emotion|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|true|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)

## Benchmarking

```bash
{
'test_accuracy': 0.938,
'test_f1': 0.937932884041714,
'test_loss': 0.1472451239824295,
'test_mem_cpu_alloc_delta': 0,
'test_mem_cpu_peaked_delta': 0,
'test_mem_gpu_alloc_delta': 0,
'test_mem_gpu_peaked_delta': 163454464,
'test_runtime': 5.0164,
'test_samples_per_second': 398.69
}
```