---
layout: model
title: Spam Classifier
author: John Snow Labs
name: classifierdl_use_spam
date: 2021-01-09
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [open_source, en]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Automatically identify messages as being regular messages or Spam.

## Predicted Entities

`spam`, `ham`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_spam_en_2.7.1_2.4_1610187019592.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_spam', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")
nlpPipeline = Pipeline(stages=[document_assembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now.')
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")
val document_classifier = ClassifierDLModel.pretrained('classifierdl_use_spam', 'en')
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val data = Seq("Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now."""]
spam_df = nlu.load('classify.spam.use').predict(text, output_level='document')
spam_df[["document", "spam"]]
```

</div>

## Results

```bash
+------------------------------------------------------------------------------------------------+------------+
|document                                                                                        |class       |
+------------------------------------------------------------------------------------------------+------------+
|Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/1234 to claim now.  | spam       |
+------------------------------------------------------------------------------------------------+------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_spam|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

This model is trained on UCI spam dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

## Benchmarking

```bash
precision    recall  f1-score   support

ham       0.99      0.99      0.99       966
spam       0.95      0.95      0.95       149

accuracy                           0.99      1115
macro avg       0.97      0.97      0.97      1115
weighted avg       0.99      0.99      0.99      1115

```