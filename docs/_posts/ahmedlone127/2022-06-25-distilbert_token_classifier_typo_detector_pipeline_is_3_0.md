---
layout: model
title: Typo Detector Pipeline for Icelandic
author: John Snow Labs
name: distilbert_token_classifier_typo_detector_pipeline
date: 2022-06-25
tags: [icelandic, typo, ner, distilbert, is, open_source]
task: Named Entity Recognition
language: is
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [distilbert_token_classifier_typo_detector_is](https://nlp.johnsnowlabs.com/2022/01/19/distilbert_token_classifier_typo_detector_is.html).

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/TYPO_DETECTOR_IS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/DistilBertForTokenClassification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_pipeline_is_4.0.0_3.0_1656119193097.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

typo_pipeline = PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "is")

typo_pipeline.annotate("Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP.")
```
```scala

val typo_pipeline = new PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "is")

typo_pipeline.annotate("Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP.")
```
</div>

## Results

```bash

+--------+---------+
|chunk   |ner_label|
+--------+---------+
|miög    |PO       |
|álykanir|PO       |
+--------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_typo_detector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|505.8 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification
- NerConverter
- Finisher