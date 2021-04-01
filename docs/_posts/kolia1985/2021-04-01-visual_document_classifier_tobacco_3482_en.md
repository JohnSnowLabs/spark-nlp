---
layout: model
title: Visual Document Classifier
author: John Snow Labs
name: visual_document_classifier_tobacco_3482
date: 2021-04-01
tags: [en, licensed]
task: Text Classification
language: en
edition: Spark NLP 3.0.0
spark_version: 3.0
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Visual Document Classifier trained on Tobacco3482 dataset.

It contains following classes: "ADVE", "Email", "Form", "Letter", "Memo", "News",  "Note", "Report", "Resume", "Scientific"

## Predicted Entities

"ADVE", "Email", "Form", "Letter", "Memo", "News",  "Note", "Report", "Resume", "Scientific"

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/visual_document_classifier_tobacco_3482_en_3.0.0_3.0_1617266158170.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pretrained_model = ("visual_document_classifier_tobacco_3482", "en", "clinical/ocr")

doc_classifier = VisualDocumentClassifier()\
    .pretrained(*pretrained_model)\
    .setInputCol("hocr")\
    .setLabelCol("label")\
    .setConfidenceCol("conf")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|visual_document_classifier_tobacco_3482|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|