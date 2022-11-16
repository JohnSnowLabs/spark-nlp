---
layout: model
title: Sarcasm Classifier
author: John Snow Labs
name: classifierdl_use_sarcasm
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

Classify if a text contains sarcasm.

## Predicted Entities

`normal`, `sarcasm`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_SARCASM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_SARCASM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_sarcasm_en_2.7.1_2.4_1610210956231.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")
use = UniversalSentenceEncoder.pretrained(lang="en") \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_sarcasm', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")
nlpPipeline = Pipeline(stages=[documentAssembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate('If I could put into words how much I love waking up at am on Tuesdays I would')
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")
val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_sarcasm", "en")
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val data = Seq("If I could put into words how much I love waking up at am on Tuesdays I would").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""If I could put into words how much I love waking up at am on Tuesdays I would"""]
sarcasm_df = nlu.load('classify.sarcasm.use').predict(text, output_level='document')
sarcasm_df[["document", "sarcasm"]]
```

</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------------+------------+
|document                                                                                                |class       |
+--------------------------------------------------------------------------------------------------------+------------+
|If I could put into words how much I love waking up at am on Tuesdays I would                           | sarcasm    |
+--------------------------------------------------------------------------------------------------------+------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_sarcasm|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

http://www.cs.utah.edu/~riloff/pdfs/official-emnlp13-sarcasm.pdf

## Benchmarking

```bash
precision    recall  f1-score   support

normal       0.98      0.89      0.93       495
sarcasm       0.60      0.91      0.73        93

accuracy                           0.89       588
macro avg       0.79      0.90      0.83       588
weighted avg       0.92      0.89      0.90       588

```