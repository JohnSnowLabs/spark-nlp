---
layout: model
title: NER Pipeline for Hindi+English
author: John Snow Labs
name: bert_token_classifier_hi_en_ner_pipeline
date: 2023-05-21
tags: [hindi, bert_token, hi, open_source]
task: Named Entity Recognition
language: hi
edition: Spark NLP 4.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on [bert_token_classifier_hi_en_ner](https://nlp.johnsnowlabs.com/2021/12/27/bert_token_classifier_hi_en_ner_hi.html).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_hi_en_ner_pipeline_hi_4.4.2_3.0_1684650604589.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_hi_en_ner_pipeline_hi_4.4.2_3.0_1684650604589.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_token_classifier_hi_en_ner_pipeline", lang = "hi")

pipeline.annotate("रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।")
```
```scala

val pipeline = new PretrainedPipeline("bert_token_classifier_hi_en_ner_pipeline", lang = "hi")

val pipeline.annotate("रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।")
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_token_classifier_hi_en_ner_pipeline", lang = "hi")

pipeline.annotate("रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।")
```
```scala
val pipeline = new PretrainedPipeline("bert_token_classifier_hi_en_ner_pipeline", lang = "hi")

val pipeline.annotate("रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।")
```
</div>

## Results

```bash
Results



+---------------------------+------------+
|chunk                      |ner_label   |
+---------------------------+------------+
|रिलायंस इंडस्ट्रीज़ लिमिटेड          |ORGANISATION|
|Reliance Industries Limited|ORGANISATION|
|भारतीय                      |PLACE       |
|मुंबई                        |PLACE       |
|महाराष्ट्र                      |PLACE       |
|Maharashtra)               |PLACE       |
|नवल टाटा                    |PERSON      |
|मुम्बई                       |PLACE       |
|Mumbai                     |PLACE       |
|टाटा समुह                    |ORGANISATION|
|भारत                       |PLACE       |
|जमशेदजी टाटा                 |PERSON      |
+---------------------------+------------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_hi_en_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|665.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- BertForTokenClassification
- NerConverter
- Finisher