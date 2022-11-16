---
layout: model
title: Sentiment Analysis of tweets Pipeline (analyze_sentimentdl_use_twitter)
author: John Snow Labs
name: analyze_sentimentdl_use_twitter
date: 2021-01-18
task: [Embeddings, Sentiment Analysis, Pipeline Public]
language: en
edition: Spark NLP 2.7.1
spark_version: 2.4
tags: [en, sentiment, pipeline, open_source]
supported: true
annotator: PipelineModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pre-trained pipeline to analyze sentiment in tweets and classify them into 'positive' and 'negative' classes using `Universal Sentence Encoder` embeddings

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SENTIMENT_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/analyze_sentimentdl_use_twitter_en_2.7.1_2.4_1610993470852.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline 
pipeline = PretrainedPipeline("analyze_sentimentdl_use_twitter", lang = "en") 

result = pipeline.fullAnnotate(["im meeting up with one of my besties tonight! Cant wait!!  - GIRL TALK!!", "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"])
```

```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val pipeline = new PretrainedPipeline("analyze_sentimentdl_use_twitter", lang = "en")
val result = pipeline.fullAnnotate("im meeting up with one of my besties tonight! Cant wait!!  - GIRL TALK!!", "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!")
```

{:.nlu-block}
```python
import nlu

text = ["""im meeting up with one of my besties tonight! Cant wait!!  - GIRL TALK!!", "is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!"""]
sentiment_df = nlu.load('en.sentiment.twitter.use').predict(text)
sentiment_df
```

</div>

## Results

```bash
|    | document                                                                                                         | sentiment   |
|---:|:---------------------------------------------------------------------------------------------------------------- |:------------|
|  0 | im meeting up with one of my besties tonight! Cant wait!!  - GIRL TALK!!                                         | positive    |
|  1 | is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!  | negative    |

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|analyze_sentimentdl_use_twitter|
|Type:|pipeline|
|Compatibility:|Spark NLP 2.7.1+|
|Edition:|Official|
|Language:|en|

## Included Models

`tfhub_use`, `sentimentdl_use_twitter`