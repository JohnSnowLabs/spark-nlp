---
layout: model
title: Emotion Detection Classifier
author: John Snow Labs
name: bert_sequence_classifier_emotion
date: 2022-01-14
tags: [bert_for_sequence, en, emotion, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned on emotion [dataset](https://huggingface.co/nateraw/bert-base-uncased-emotion), leveraging `Bert` embeddings and `BertForSequenceClassification` for text classification purposes.

## Predicted Entities

`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_emotion_en_3.3.4_3.0_1642152012549.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_emotion_en_3.3.4_3.0_1642152012549.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
.pretrained('bert_sequence_classifier_emotion', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([["What do you mean? Are you kidding me?"]]).toDF("text")

result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_emotion", "en")
.setInputCols(Array("document", "token"))
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["What do you mean? Are you kidding me?"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.emotion.bert").predict("""What do you mean? Are you kidding me?""")
```

</div>

## Results

```bash
['anger']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_emotion|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## Data Source

[https://huggingface.co/datasets/viewer/?dataset=emotion](https://huggingface.co/datasets/viewer/?dataset=emotion)

## Benchmarking

NOTE: The author didn't share Precision / Recall / F1, only Validation Accuracy was shared as [Evaluation Results](https://huggingface.co/nateraw/bert-base-uncased-emotion#eval-results).

```bash
Validation Accuracy: 0.931 
```
