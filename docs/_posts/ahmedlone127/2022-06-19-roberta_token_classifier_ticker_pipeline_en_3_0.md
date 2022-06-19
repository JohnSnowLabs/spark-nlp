---
layout: model
title: Pipeline to Detect Alias in Financial texts
author: John Snow Labs
name: roberta_token_classifier_ticker_pipeline
date: 2022-06-19
tags: [ticker, roberta, pipeline, finance, ner, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [roberta_token_classifier_ticker](https://nlp.johnsnowlabs.com/2021/12/27/roberta_token_classifier_ticker_en.html) model.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TICKER/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_ticker_pipeline_en_4.0.0_3.0_1655653865169.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


pipeline = PretrainedPipeline("roberta_token_classifier_ticker_pipeline", lang = "en")

pipeline.annotate("I am going to buy 100 shares of MFST tomorrow instead of AAPL.")
```
```scala


val pipeline = new PretrainedPipeline("roberta_token_classifier_ticker_pipeline", lang = "en")

pipeline.annotate("I am going to buy 100 shares of MFST tomorrow instead of AAPL.")
```
</div>

## Results

```bash


+-----+---------+
|chunk|ner_label|
+-----+---------+
|MFST |TICKER   |
|AAPL |TICKER   |
+-----+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_ticker_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|465.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- RoBertaForTokenClassification
- NerConverter
- Finisher