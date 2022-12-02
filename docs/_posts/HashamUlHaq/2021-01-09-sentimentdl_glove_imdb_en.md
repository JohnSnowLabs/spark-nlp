---
layout: model
title: Sentiment Analysis of IMDB Reviews
author: John Snow Labs
name: sentimentdl_glove_imdb
date: 2021-01-09
task: Sentiment Analysis
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [open_source, en, sentiment]
supported: true
annotator: SentimentDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Classify IMDB reviews in negative and positive categories.

## Predicted Entities

`neg`, `pos`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimentdl_glove_imdb_en_2.7.1_2.4_1610208660282.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
embeddings = WordEmbeddingsModel().pretrained("glove_100d)\
.setInputCols(['document','tokens'])\
.setOutputCol('word_embeddings')
sentence_embeddings = SentenceEmbeddings() \
.setInputCols(["document", "word_embeddings"]) \
.setOutputCol("sentence_embeddings") \
.setPoolingStrategy("AVERAGE")
classifier = SentimentDLModel().pretrained('sentimentdl_glove_imdb')\
.setInputCols(["sentence_embeddings"])\
.setOutputCol("sentiment")
nlp_pipeline = Pipeline(stages=[document_assembler, sentencer, tokenizer, embeddings, sentence_embeddings, classifier])
l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate('Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!')
```

```scala
...
val embeddings = WordEmbeddingsModel().pretrained("glove_100d)
.setInputCols(Array('document','tokens'))
.setOutputCol('word_embeddings')
val sentence_embeddings = SentenceEmbeddings()
.setInputCols(Array("document", "word_embeddings"))
.setOutputCol("sentence_embeddings")
.setPoolingStrategy("AVERAGE")
val classifier = SentimentDLModel().pretrained('sentimentdl_glove_imdb')
.setInputCols(Array("sentence_embeddings"))
.setOutputCol("sentiment")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentencer, tokenizer, embeddings, sentence_embeddings, classifier))
val data = Seq("Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!").toDF("text")
val result = pipeline.fit(data).transform(data)
```



{:.nlu-block}
```python
import nlu
nlu.load("en.sentiment.imdb.glove").predict(""")
nlp_pipeline = Pipeline(stages=[document_assembler, sentencer, tokenizer, embeddings, sentence_embeddings, classifier])
l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF(""")
```

</div>

## Results

```bash
|    | document                                                                                                 | sentiment     |
|---:|---------------------------------------------------------------------------------------------------------:|--------------:|
|    | Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the  |               |
|  0 | film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music   | positive      |
|    | was rad! Horror and sword fight freaks,buy this movie now!                                               |               |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentimentdl_glove_imdb|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[sentiment]|
|Language:|en|
|Dependencies:|glove_840B_300|

## Data Source

https://ai.stanford.edu/~amaas/data/sentiment/

## Benchmarking

```bash
precision    recall  f1-score   support

neg       0.85      0.85      0.85     12500
pos       0.87      0.83      0.85     12500

accuracy                           0.84     25000
macro avg       0.86      0.84      0.85     25000
weighted avg       0.86      0.84      0.85     25000

```