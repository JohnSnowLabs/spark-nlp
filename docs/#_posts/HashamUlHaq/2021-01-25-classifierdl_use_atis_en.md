---
layout: model
title: Intent Classification for Airline Traffic Information System queries (ATIS dataset)
author: John Snow Labs
name: classifierdl_use_atis
date: 2021-01-25
task: Text Classification
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [en, classifier, open_source]
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify user questions into 5 categories of an airline traffic information system.

## Predicted Entities

`atis_abbreviation`, `atis_airfare`, `atis_airline`, `atis_flight`, `atis_ground_service`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_use_atis_en_2.7.1_2.4_1611572512585.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

document_classifier = ClassifierDLModel.pretrained('classifierdl_use_atis', 'en') \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

nlpPipeline = Pipeline(stages=[document_assembler, use, document_classifier])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(['what is the price of flight from newyork to washington', 'how many flights does twa have in business class'])

```



{:.nlu-block}
```python
import nlu
nlu.load("en.classify.intent.airline").predict("""what is the price of flight from newyork to washington""")
```

</div>

## Results

```bash
+-------------------------------------------------------------------+----------------+
| document                                                          | class          |
+-------------------------------------------------------------------+----------------+
| what is the price of flight from newyork to washington			| atis_airfare   |
| how many flights does twa have in business class					| atis_quantity  |
+-------------------------------------------------------------------+----------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_use_atis|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Dependencies:|tfhub_use|

## Data Source

This model is trained on data obtained from https://www.kaggle.com/hassanamin/atis-airlinetravelinformationsystem

## Benchmarking

```bash
precision    recall  f1-score   support

atis_abbreviation       1.00      1.00      1.00        33
atis_airfare       0.60      0.96      0.74        48
atis_airline       0.69      0.89      0.78        38
atis_flight       0.99      0.93      0.96       632
atis_ground_service       1.00      1.00      1.00        36

accuracy                           0.93       787
macro avg       0.86      0.96      0.90       787
weighted avg       0.95      0.93      0.94       787

```