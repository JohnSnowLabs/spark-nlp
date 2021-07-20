---
layout: model
title: Visual Document NER with SROIE
author: John Snow Labs
name: visual_document_NER_SROIE0518
date: 2021-05-17
tags: [en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP 2.5.5
spark_version: 3.0
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is fine tuned on SROIE dataset with LayoutLM pre-trained model.

## Predicted Entities

O
B-DATE
B-COMPANY
B-TOTAL

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/ocr/visual_document_NER_SROIE0518_en_2.5.5_3.0_1621295626060.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

by feeding hocr input to this model and then it should output the related label per word or token

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
doc_ner = VisualDocumentNer()\
            .pretrained("visual_document_NER_SROIE0518", "en", "public/ocr/models") \
            .setInputCol("hocr")\
            .setLabelCol("label")
```
```scala
    val visualDocumentNER = VisualDocumentNER
      .pretrained(testSparkModel, "en", "public/ocr/models")
      .setInputCol("hocr")
```
</div>

## Results

```bash
+----------+----------+---------+
|word      |token     |label    |
+----------+----------+---------+
|RESTAURANT|restaurant|B-COMPANY|
|JTJ       |jtj       |B-COMPANY|
|JTJ       |jtj       |B-COMPANY|
|JTJ       |jtj       |B-COMPANY|
|FOODS     |foods     |B-COMPANY|
|SDN       |sdn       |B-COMPANY|
|SDN       |sdn       |B-COMPANY|
|BHD       |bhd       |B-COMPANY|
|BHD       |bhd       |B-COMPANY|
|6/1/2018  |6         |B-DATE   |
|6/1/2018  |1         |B-DATE   |
|6/1/2018  |2018      |B-DATE   |
|10.30     |10        |B-TOTAL  |
|10.30     |30        |B-TOTAL  |
|10.30     |10        |B-TOTAL  |
|10.30     |30        |B-TOTAL  |
+----------+----------+---------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|visual_document_NER_SROIE0518|
|Type:|ocr|
|Compatibility:|Spark NLP 2.5.5+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

SROIE