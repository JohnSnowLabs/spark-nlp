---
layout: model
title: Identify intent in general text - SNIPS dataset
author: John Snow Labs
name: classifierdl_use_snips
date: 2021-02-15
task: Text Classification
language: en
edition: Spark NLP 2.7.3
spark_version: 2.4
tags: [open_source, classifier, en]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Understand general commands and recognise the intent.

## Predicted Entities

`AddToPlaylist`, `BookRestaurant`, `GetWeather`, `PlayMusic`, `RateBook`, `SearchCreativeWork`, `SearchScreeningEvent`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_CLS_SNIPS){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_snips_en_2.7.3_2.4_1613416966282.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/classifierdl_use_snips_en_2.7.3_2.4_1613416966282.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

embeddings = UniversalSentenceEncoder.pretrained('tfhub_use', lang="en") \
.setInputCols(["document"])\
.setOutputCol("sentence_embeddings")

classifier = ClassifierDLModel.pretrained('classifierdl_use_snips').setInputCols(['sentence_embeddings']).setOutputCol('class')

nlp_pipeline = Pipeline(stages=[document_assembler, embeddings, classifier])

l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate(["i want to bring six of us to a bistro in town that serves hot chicken sandwich that is within the same area", "show weather forcast for t h  stone memorial st  joseph peninsula state park on one hour from now"])
```
```scala
...
val embeddings = UniversalSentenceEncoder.pretrained("tfhub_use", lang="en") \
.setInputCols(Array("document"))\
.setOutputCol("sentence_embeddings")

val classifier = ClassifierDLModel.pretrained("classifierdl_use_snips", "en").setInputCols(Array("sentence_embeddings")).setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, embeddings, classifier))

val data = Seq("i want to bring six of us to a bistro in town that serves hot chicken sandwich that is within the same area", "show weather forcast for t h  stone memorial st  joseph peninsula state park on one hour from now").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.snips").predict("""i want to bring six of us to a bistro in town that serves hot chicken sandwich that is within the same area""")
```

</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------------+----------------+
| document                                                        										        | label          |
+---------------------------------------------------------------------------------------------------------------+----------------+
| i want to bring six of us to a bistro in town that serves hot chicken sandwich that is within the same area   | BookRestaurant |
| show weather forcast for t h  stone memorial st  joseph peninsula state park on one hour from now				| GetWeather	 |
+---------------------------------------------------------------------------------------------------------------+----------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_snips|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|

## Data Source

This model is trained on the NLU Benchmark, SNIPS dataset https://github.com/MiuLab/SlotGated-SLU

## Benchmarking

```bash
precision    recall  f1-score   support

AddToPlaylist       0.98      0.97      0.97       124
BookRestaurant       0.98      0.99      0.98        92
GetWeather       1.00      0.98      0.99       104
PlayMusic       0.85      0.95      0.90        86
RateBook       1.00      1.00      1.00        80
SearchCreativeWork       0.82      0.84      0.83       107
SearchScreeningEvent       0.95      0.85      0.90       107

accuracy                           0.94       700
macro avg       0.94      0.94      0.94       700
weighted avg       0.94      0.94      0.94       700
```
