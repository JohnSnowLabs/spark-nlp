---
layout: model
title: Sentiment Analysis of IMDB Reviews Pipeline (analyze_sentimentdl_use_imdb)
author: John Snow Labs
name: analyze_sentimentdl_use_imdb
date: 2021-01-15
task: [Embeddings, Sentiment Analysis, Pipeline Public]
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [en, pipeline, sentiment]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pre-trained pipeline to classify IMDB reviews in `neg` and `pos` classes using `tfhub_use` embeddings.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/analyze_sentimentdl_use_imdb_en_2.7.1_2.4_1610723836151.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/analyze_sentimentdl_use_imdb_en_2.7.1_2.4_1610723836151.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("analyze_sentimentdl_use_imdb", lang = "en") 
result = pipeline.fullAnnotate("Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("analyze_sentimentdl_use_imdb", lang = "en")
val result = pipeline.fullAnnotate("Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!")
```

{:.nlu-block}
```python
import nlu

text = ["""Demonicus is a movie turned into a video game! I just love the story and the things that goes on in the film.It is a B-film ofcourse but that doesn`t bother one bit because its made just right and the music was rad! Horror and sword fight freaks,buy this movie now!"""]
sentiment_df = nlu.load('en.sentiment.imdb.use').predict(text, output_level='sentence')
sentiment_df
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
|Model Name:|analyze_sentimentdl_use_imdb|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.1+|
|Edition:|Official|
|Language:|en|

## Included Models

`tfhub_use`, `sentimentdl_use_imdb`