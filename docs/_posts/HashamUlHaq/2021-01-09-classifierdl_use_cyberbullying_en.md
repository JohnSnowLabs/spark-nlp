---
layout: model
title: Cyberbullying Classifier
author: John Snow Labs
name: classifierdl_use_cyberbullying
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

Identify Racism, Sexism or Neutral tweets.

## Predicted Entities

`neutral`, `racism`, `sexism`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN_CYBERBULLYING/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN_CYBERBULLYING.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_cyberbullying_en_2.7.1_2.4_1610188083627.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
document_classifier = ClassifierDLModel.pretrained('classifierdl_use_cyberbullying', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")
nlpPipeline = Pipeline(stages=[document_assembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate('@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked')
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")
val use = UniversalSentenceEncoder.pretrained(lang="en")
.setInputCols(Array("document"))
.setOutputCol("sentence_embeddings")
val document_classifier = ClassifierDLModel.pretrained("classifierdl_use_cyberbullying", "en")
.setInputCols(Array("document", "sentence_embeddings"))
.setOutputCol("class")
val pipeline = new Pipeline().setStages(Array(documentAssembler, use, document_classifier))

val data = Seq("@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked"""]
cyberbull_df = nlu.load('classify.cyberbullying.use').predict(text, output_level='document')
cyberbull_df[["document", "cyberbullying"]]
```

{:.nlu-block}
```python

```



{:.nlu-block}
```python
import nlu
nlu.load("en.classify.cyberbullying").predict("""@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked""")
```



{:.nlu-block}
```python
import nlu
nlu.load("en.classify.cyberbullying").predict("""@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked""")
```

</div>

## Results

```bash
+--------------------------------------------------------------------------------------------------------+------------+
|document                                                                                                |class       |
+--------------------------------------------------------------------------------------------------------+------------+
|@geeky_zekey Thanks for showing again that blacks are the biggest racists. Blocked.                     | racism     |
+--------------------------------------------------------------------------------------------------------+------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_cyberbullying|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

This model is trained on cyberbullying detection dataset. https://raw.githubusercontent.com/dhavalpotdar/cyberbullying-detection/master/data/data/data.csv

## Benchmarking

```bash
precision    recall  f1-score   support

neutral       0.72      0.76      0.74       700
racism       0.89      0.94      0.92       773
sexism       0.82      0.71      0.76       622

accuracy                           0.81      2095
macro avg       0.81      0.80      0.80      2095
weighted avg       0.81      0.81      0.81      2095
```