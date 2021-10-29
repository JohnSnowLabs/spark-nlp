---
layout: model
title: Sentiment Analysis Pipeline for German texts
author: John Snow Labs
name: classifierdl_bert_sentiment_pipeline
date: 2021-09-28
tags: [sentiment, de, pipeline, open_source]
task: Sentiment Analysis
language: de
edition: Spark NLP 3.3.0
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline identifies the sentiments (positive or negative) in German texts.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/SENTIMENT_DE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/CLASSIFICATION_De_SENTIMENT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/classifierdl_bert_sentiment_pipeline_de_3.3.0_2.4_1632832830977.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "de")
result = pipeline.annotate("Spiel und Meisterschaft nicht spannend genug? Muss man jetzt den Videoschiedsrichter kontrollieren? Ich bin entsetzt...dachte der darf nur bei krassen Fehlentscheidungen ran. So macht der Fussball keinen Spass mehr.")
```
```scala
val pipeline = new PretrainedPipeline("classifierdl_bert_sentiment_pipeline", lang = "de")

val result = pipeline.fullAnnotate("Spiel und Meisterschaft nicht spannend genug? Muss man jetzt den Videoschiedsrichter kontrollieren? Ich bin entsetzt...dachte der darf nur bei krassen Fehlentscheidungen ran. So macht der Fussball keinen Spass mehr.")(0)
```
</div>

## Results

```bash
['NEGATIVE']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|classifierdl_bert_sentiment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.3.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|

## Included Models

- DocumentAssembler
- BertSentenceEmbeddings
- ClassifierDLModel