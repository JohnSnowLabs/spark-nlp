---
layout: model
title: BERT Sequence Classification - German Sentiment Analysis (bert_sequence_classifier_sentiment)
author: John Snow Labs
name: bert_sequence_classifier_sentiment
date: 2021-11-03
tags: [german, de, bert, sentiment, sequence_classification, open_source]
task: Text Classification
language: de
edition: Spark NLP 3.3.2
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

# German Sentiment Classification with Bert

This model was trained for sentiment classification of German language texts. To achieve the best results all model inputs needs to be preprocessed with the same procedure, that was applied during the training. To simplify the usage of the model,
we provide a Python package that bundles the code need for the preprocessing and inferencing.

The model uses the Googles Bert architecture and was trained on 1.834 million German-language samples. The training data contains texts from various domains like Twitter, Facebook and movie, app and hotel reviews.

You can find more information about the dataset and the training process in the [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.202.pdf).

## Cite

Please cite this paper if you found this useful:

```
@InProceedings{guhr-EtAl:2020:LREC,
  author    = {Guhr, Oliver  and  Schumann, Anne-Kathrin  and  Bahrmann, Frank  and  BÃ¶hme, Hans Joachim},
  title     = {Training a Broad-Coverage German Sentiment Classification Model for Dialog Systems},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {1620--1625},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.202}
}
```

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_sentiment_de_3.3.2_3.0_1635941038799.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
      .pretrained('bert_sequence_classifier_sentiment', 'de') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    sequenceClassifier
])

example = spark.createDataFrame([['Mit keinem guten Ergebniss']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_sentiment", "de")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Mit keinem guten Ergebniss").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sentiment|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|de|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/oliverguhr/german-sentiment-bert](https://huggingface.co/oliverguhr/german-sentiment-bert)

## Benchmarking

```bash
| Dataset                                                      | F1 micro Score |
| :----------------------------------------------------------- | -------------: |
| [holidaycheck](https://github.com/oliverguhr/german-sentiment) |         0.9568 |
| [scare](https://www.romanklinger.de/scare/)                  |         0.9418 |
| [filmstarts](https://github.com/oliverguhr/german-sentiment) |         0.9021 |
| [germeval](https://sites.google.com/view/germeval2017-absa/home) |         0.7536 |
| [PotTS](https://www.aclweb.org/anthology/L16-1181/)          |         0.6780 |
| [emotions](https://github.com/oliverguhr/german-sentiment)  |         0.9649 |
| [sb10k](https://www.spinningbytes.com/resources/germansentiment/) |         0.7376 |
| [Leipzig Wikipedia Corpus 2016](https://wortschatz.uni-leipzig.de/de/download/german) |         0.9967 |
| all                                                          |         0.9639 |
```