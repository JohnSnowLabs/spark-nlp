---
layout: model
title: Typo Detector Pipeline for English
author: ahmedlone127
name: distilbert_token_classifier_typo_detector_pipeline
date: 2022-06-14
tags: [ner, bert, bert_for_token, typo, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: false
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [distilbert_token_classifier_typo_detector](https://nlp.johnsnowlabs.com/2022/01/19/distilbert_token_classifier_typo_detector_en.html).

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/TYPO_DETECTOR_EN/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/DistilBertForTokenClassification.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/ahmedlone127/distilbert_token_classifier_typo_detector_pipeline_en_4.0.0_3.0_1655212406234.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/ahmedlone127/distilbert_token_classifier_typo_detector_pipeline_en_4.0.0_3.0_1655212406234.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

typo_pipeline = PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "en")

typo_pipeline.annotate("He had also stgruggled with addiction during his tine in Congress.")
```
```scala

val typo_pipeline = new PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "en")

typo_pipeline.annotate("He had also stgruggled with addiction during his tine in Congress.")
```
</div>

## Results

```bash

+----------+---------+
|chunk     |ner_label|
+----------+---------+
|stgruggled|PO       |
|tine      |PO       |
+----------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_typo_detector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Community|
|Language:|en|
|Size:|244.2 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification
- NerConverter
- Finisher