---
layout: model
title: Fake News Classifier
author: John Snow Labs
name: classifierdl_use_fakenews
date: 2021-01-09
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [open_source, en, classifier]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Determine if news articles are Real or Fake.

## Predicted Entities

`REAL`, `FAKE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_EN_FAKENEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_EN_FAKENEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_fakenews_en_2.7.1_2.4_1610187399147.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_fakenews', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")
nlpPipeline = Pipeline(stages=[document_assembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate('Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton')
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")
val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_fakenews", "en")
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val data = Seq("Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton"""]
fake_df = nlu.load('classify.fakenews.use').predict(text, output_level='document')
fake_df[["document", "fakenews"]]
```

</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------------+------------+
|document                                                                                                |class       |
+--------------------------------------------------------------------------------------------------------+------------+
|Donald Trump a KGB Spy? 11/02/2016 In today’s video, Christopher Greene of AMTV reports Hillary Clinton | FAKE       |
+--------------------------------------------------------------------------------------------------------+------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_fakenews|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

This model is trained on the fake new classification challenge. https://raw.githubusercontent.com/joolsa/fake_real_news_dataset/master/fake_or_real_news.csv.zip

## Benchmarking

```bash
precision    recall  f1-score   support

FAKE       0.86      0.89      0.88       626
REAL       0.89      0.86      0.87       634

accuracy                           0.87      1260
macro avg       0.88      0.87      0.87      1260
weighted avg       0.88      0.87      0.87      1260
```