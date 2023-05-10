---
layout: model
title: Sentiment Analysis for Thai (sentiment_jager_use)
author: John Snow Labs
name: sentiment_jager_use
date: 2021-01-14
task: Sentiment Analysis
language: th
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [sentiment, th, open_source]
supported: true
annotator: SentimentDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Analyze sentiment in reviews by classifying them as `positive` and `negative`. When the sentiment probability is below a customizable threshold (default to `0.6`)  then resulting document will be labeled as `neutral`. This model is trained using the multilingual `UniversalSentenceEncoder` sentence embeddings, and uses DL approach to classify the sentiments.

## Predicted Entities

`positive`, `negative`, `neutral`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentiment_jager_use_th_2.7.1_2.4_1610586390122.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentiment_jager_use_th_2.7.1_2.4_1610586390122.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use in the pipeline with the pretrained multi-language `UniversalSentenceEncoder` annotator `tfhub_use_multi_lg`.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained("tfhub_use_multi_lg", "xx") \
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")
sentimentdl = SentimentDLModel.pretrained("sentiment_jager_use", "th")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")
pipeline = Pipeline(stages = [document_assembler, use, sentimentdl])

example = spark.createDataFrame([['เเพ้ตอนnctโผล่มาตลอดเลยค่ะเเอด5555555']], ["text"])
result = pipeline.fit(example).transform(example)
```

```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val use = UniversalSentenceEncoder.pretrained("tfhub_use_multi_lg", "xx")
    .setInputCols(Array("document")
    .setOutputCol("sentence_embeddings")
val sentimentdl = SentimentDLModel.pretrained("sentiment_jager_use", "th")
    .setInputCols(Array("sentence_embeddings"))
    .setOutputCol("sentiment")
val pipeline = new Pipeline().setStages(Array(document_assembler, use, sentimentdl))
val data = Seq("เเพ้ตอนnctโผล่มาตลอดเลยค่ะเเอด5555555").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""เเพ้ตอนnctโผล่มาตลอดเลยค่ะเเอด5555555"""]
sentiment_df = nlu.load('th.classify.sentiment').predict(text)
sentiment_df
```

</div>

## Results

```bash
+-------------------------------------+----------+
|text                                 |result    |
+-------------------------------------+----------+
|เเพ้ตอนnctโผล่มาตลอดเลยค่ะเเอด5555555  |[positive] |
+-------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentiment_jager_use|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[sentiment]|
|Language:|th|

## Data Source

The model was trained on the custom corpus from [Jager V3](https://github.com/JagerV3/sentiment_analysis_thai).

## Benchmarking

```bash
| sentiment    | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| negative     | 0.94      | 0.99   | 0.96     | 82      |
| positive     | 0.97      | 0.87   | 0.92     | 38      |
| accuracy     |           |        | 0.95     | 120     |
| macro avg    | 0.96      | 0.93   | 0.94     | 120     |
| weighted avg | 0.95      | 0.95   | 0.95     | 120     |
```
