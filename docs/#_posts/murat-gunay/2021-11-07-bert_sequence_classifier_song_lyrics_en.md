---
layout: model
title: BERT Sequence Classifier - Classify the Music Genre
author: John Snow Labs
name: bert_sequence_classifier_song_lyrics
date: 2021-11-07
tags: [song, lyrics, en, bert_for_sequence_classification, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: BertForSequenceClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face-models` and it classifies the music genre into 6 classes.

## Predicted Entities

`Dance`, `Heavy Metal`, `Hip Hop`, `Indie`, `Pop`, `Rock`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_song_lyrics_en_3.3.2_2.4_1636283685615.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
.pretrained('bert_sequence_classifier_song_lyrics', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setCaseSensitive(True) \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([["""Because you need me Every single day Trying to find me But you don't know why Trying to find me again But you don't know how Trying to find me again Every single day"""]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = Tokenizer() 
.setInputCols("document") 
.setOutputCol("token")

val tokenClassifier = BertForSequenceClassification("bert_sequence_classifier_song_lyrics", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setCaseSensitive(true)
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["""Because you need me Every single day Trying to find me But you don't know why Trying to find me again But you don't know how Trying to find me again Every single day"""].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.song_lyrics").predict("""Because you need me Every single day Trying to find me But you don't know why Trying to find me again But you don't know how Trying to find me again Every single day""")
```

</div>

## Results

```bash
['Rock']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_song_lyrics|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/juliensimon/autonlp-song-lyrics-18753417](https://huggingface.co/juliensimon/autonlp-song-lyrics-18753417)

## Benchmarking

```bash
+--------------------+----------+
| Validation Metrics | Score    |
+--------------------+----------+
| Loss               | 0.906597 |
| Accuracy           | 0.668027 |
| Macro F1           | 0.538484 |
| Micro F1           | 0.668027 |
| Weighted F1        | 0.64147  |
| Macro Precision    | 0.67444  |
| Micro Precision    | 0.668027 |
| Weighted Precision | 0.663409 |
| Macro Recall       | 0.50784  |
| Micro Recall       | 0.668027 |
| Weighted Recall    | 0.668027 |
+--------------------+----------+

```
