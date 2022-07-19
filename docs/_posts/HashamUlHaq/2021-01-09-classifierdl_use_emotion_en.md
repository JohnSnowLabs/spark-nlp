---
layout: model
title: Emotion Detection Classifier
author: John Snow Labs
name: classifierdl_use_emotion
date: 2021-01-09
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [open_source, en, classifier]
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Automatically identify Joy, Surprise, Fear, Sadness emotions in Tweets.

## Predicted Entities

`surprise`, `sadness`, `fear`, `joy`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_EMOTION/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_EMOTION.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_emotion_en_2.7.1_2.4_1610190563302.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")
use = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_emotion', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")
nlpPipeline = Pipeline(stages=[document_assembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate('@Mira I just saw you on live t.v!!')
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")
val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_emotion", "en")
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val data = Seq("@Mira I just saw you on live t.v!!").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""@Mira I just saw you on live t.v!!"""]
emotion_df = nlu.load('en.classify.emotion.use').predict(text, output_level='document')
emotion_df[["document", "emotion"]]
```

</div>

## Results

```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|@Mira I just saw you on live t.v!!                                                              | joy        |
+------------------------------------------------------------------------------------------------+------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_emotion|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

This model is trained on multiple datasets inlcuding youtube comments, twitter and ISEAR dataset.

## Benchmarking

```bash
fear       0.78      0.67      0.72      2253
joy       0.71      0.68      0.69      3000
sadness       0.69      0.73      0.71      3075
surprise       0.67      0.73      0.70      3067

accuracy                           0.71     11395
macro avg       0.71      0.70      0.71     11395
weighted avg       0.71      0.71      0.71     11395
```