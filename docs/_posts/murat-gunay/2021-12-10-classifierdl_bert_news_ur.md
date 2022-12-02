---
layout: model
title: News Classifier for Urdu texts
author: John Snow Labs
name: classifierdl_bert_news
date: 2021-12-10
tags: [urdu, news, classifier, ur, open_source]
task: Text Classification
language: ur
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
annotator: ClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify Urdu news into 7 categories.

## Predicted Entities

`business`, `entertainment`, `health`, `inland`, `science`, `sports`, `weird_news`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_UR_NEWS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_UR_NEWS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_news_ur_3.3.0_2.4_1639125233132.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
.setInputCol("news") \
.setOutputCol("document")

embeddings = BertSentenceEmbeddings.pretrained("labse", "xx") \
.setInputCols("document") \
.setOutputCol("sentence_embeddings")

classifierdl = ClassifierDLModel.pretrained("classifierdl_bert_news", "ur") \
.setInputCols(["document", "sentence_embeddings"]) \
.setOutputCol("class")

urdu_news_pipeline = Pipeline(stages=[document_assembler, embeddings, classifierdl])
light_pipeline = LightPipeline(urdu_news_pipeline.fit(spark.createDataFrame([['']]).toDF("news")))

result = light_pipeline.annotate("گزشتہ ہفتے ایپل کے حصص میں 11 فیصد اضافہ ہوا ہے۔")
result["class"]
```
```scala
val document = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val embeddings = BertSentenceEmbeddings
.pretrained("lanse", "xx") 
.setInputCols("document")
.setOutputCol("sentence_embeddings")

val document_classifier = ClassifierDLModel.pretrained("classifierdl_bert_news", "ur") 
.setInputCols(Array("document", "sentence_embeddings")) 
.setOutputCol("class")

val nlpPipeline = new Pipeline().setStages(Array(document, embeddings, document_classifier))
val light_pipeline = LightPipeline(nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text")))
val result = light_pipeline.annotate("گزشتہ ہفتے ایپل کے حصص میں 11 فیصد اضافہ ہوا ہے۔")

```


{:.nlu-block}
```python
import nlu
nlu.load("ur.classify.news").predict("""گزشتہ ہفتے ایپل کے حصص میں 11 فیصد اضافہ ہوا ہے۔""")
```

</div>

## Results

```bash
['business']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_news|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|ur|
|Size:|23.6 MB|

## Data Source

Combination of multiple open source data sets.

## Benchmarking

```bash
label  precision    recall  f1-score   support
business       0.83      0.86      0.85      2365
entertainment       0.87      0.85      0.86      3081
health       0.68      0.67      0.68       430
inland       0.80      0.82      0.81      3964
science       0.62      0.60      0.61       558
sports       0.88      0.89      0.89      4022
weird_news       0.60      0.54      0.57       826
accuracy          -         -      0.82     15246
macro-avg       0.76      0.75      0.75     15246
weighted-avg       0.82      0.82      0.82     15246
```
