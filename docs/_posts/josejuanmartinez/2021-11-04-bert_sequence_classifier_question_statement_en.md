---
layout: model
title: Bert for Sequence Classification (Question vs Statement)
author: John Snow Labs
name: bert_sequence_classifier_question_statement
date: 2021-11-04
tags: [question, statement, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Trained to add sentence classifying capabilities to distinguish between Question vs Statements.

This model was imported from Hugging Face (https://huggingface.co/shahrukhx01/question-vs-statement-classifier), and trained based on Haystack (https://github.com/deepset-ai/haystack/issues/611).

## Predicted Entities

`question`, `statement`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_question_statement_en_3.3.2_3.0_1636038134936.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_question_statement_en_3.3.2_3.0_1636038134936.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained() \
.setInputCols(["document"]) \
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")

seq = BertForSequenceClassification.pretrained('bert_sequence_classifier_question_statement', 'en')\
.setInputCols(["token", "sentence"])\
.setOutputCol("label")\
.setCaseSensitive(True)

pipeline = Pipeline(stages = [
documentAssembler,
sentenceDetector,
tokenizer,
seq])

test_sentences = ["""What feature in your car did you not realize you had until someone else told you about it?
Years ago, my Dad bought me a cute little VW Beetle. The first day I had it, me and my BFF were sitting in my car looking at everything.
When we opened the center console, we had quite the scare. Inside was a hollowed out, plastic phallic looking object with tiny spikes on it.
My friend and I literally screamed in horror. It was clear to us that somehow someone left their “toy” in my new car! We were shook, as they say.
This was my car, I had to do something. So, I used a pen to pick up the nasty looking thing and threw it out.
We freaked out about how gross it was and then we forgot about it… until my Dad called me.
My Dad said: How’s the new car? Have you seen the flower holder in the center console?
To summarize, we thought a flower vase was an XXX item…
In our defense, this is a picture of a VW Beetle flower holder."""]

import pandas as pd
data=spark.createDataFrame(pd.DataFrame({'text': test_sentences}))
res = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols("sentence")
.setOutputCol("token")

val seq = BertForSequenceClassification.pretrained("bert_sequence_classifier_question_statement", "en")
.setInputCols(Array("token", "sentence"))
.setOutputCol("label")
.setCaseSensitive(True)

val pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentenceDetector,
tokenizer,
seq))

val test_sentences = "What feature in your car did you not realize you had until someone else told you about it?
Years ago, my Dad bought me a cute little VW Beetle. The first day I had it, me and my BFF were sitting in my car looking at everything.
When we opened the center console, we had quite the scare. Inside was a hollowed out, plastic phallic looking object with tiny spikes on it.
My friend and I literally screamed in horror. It was clear to us that somehow someone left their “toy” in my new car! We were shook, as they say.
This was my car, I had to do something. So, I used a pen to pick up the nasty looking thing and threw it out.
We freaked out about how gross it was and then we forgot about it… until my Dad called me.
My Dad said: How’s the new car? Have you seen the flower holder in the center console?
To summarize, we thought a flower vase was an XXX item…
In our defense, this is a picture of a VW Beetle flower holder."

val example = Seq(test_sentences).toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.question_vs_statement").predict("""What feature in your car did you not realize you had until someone else told you about it?
Years ago, my Dad bought me a cute little VW Beetle. The first day I had it, me and my BFF were sitting in my car looking at everything.
When we opened the center console, we had quite the scare. Inside was a hollowed out, plastic phallic looking object with tiny spikes on it.
My friend and I literally screamed in horror. It was clear to us that somehow someone left their “toy” in my new car! We were shook, as they say.
This was my car, I had to do something. So, I used a pen to pick up the nasty looking thing and threw it out.
We freaked out about how gross it was and then we forgot about it… until my Dad called me.
My Dad said: How’s the new car? Have you seen the flower holder in the center console?
To summarize, we thought a flower vase was an XXX item…
In our defense, this is a picture of a VW Beetle flower holder.""")
```

</div>

## Results

```bash
+------------------------------------------------------------------------------------------+---------+
|                                                                                  sentence|    label|
+------------------------------------------------------------------------------------------+---------+
|What feature in your car did you not realize you had until someone else told you about it?| question|
|                                      Years ago, my Dad bought me a cute little VW Beetle.|statement|
|       The first day I had it, me and my BFF were sitting in my car looking at everything.|statement|
|                                When we opened the center console, we had quite the scare.|statement|
|         Inside was a hollowed out, plastic phallic looking object with tiny spikes on it.|statement|
|                                             My friend and I literally screamed in horror.|statement|
|                   It was clear to us that somehow someone left their “toy” in my new car!|statement|
|                                                               We were shook, as they say.|statement|
|                                                   This was my car, I had to do something.|statement|
|                     So, I used a pen to pick up the nasty looking thing and threw it out.|statement|
|We freaked out about how gross it was and then we forgot about it… until my Dad called me.|statement|
|                                                           My Dad said: How’s the new car?| question|
|                                    Have you seen the flower holder in the center console?| question|
|                                   To summarize, we thought a flower vase was an XXX item…|statement|
|                           In our defense, this is a picture of a VW Beetle flower holder.|statement|
+------------------------------------------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_question_statement|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|

## Data Source

https://github.com/deepset-ai/haystack/issues/611

## Benchmarking

```bash
Extracted from https://github.com/deepset-ai/haystack/issues/611
precision    recall  f1-score   support

statement        0.94      0.94      0.94     16105
question        0.96      0.96      0.96     26198

accuracy                           0.95     42303
macro avg       0.95      0.95      0.95     42303
weighted avg       0.95      0.95      0.95     42303
```
