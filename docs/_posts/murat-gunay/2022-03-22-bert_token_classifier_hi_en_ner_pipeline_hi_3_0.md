---
layout: model
title: NER Pipeline for Hindi+English
author: John Snow Labs
name: bert_token_classifier_hi_en_ner_pipeline
date: 2022-03-22
tags: [hindi, bert_token, hi, open_source]
task: Named Entity Recognition
language: hi
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on [bert_token_classifier_hi_en_ner](https://nlp.johnsnowlabs.com/2021/12/27/bert_token_classifier_hi_en_ner_hi.html).

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_HINDI_ENGLISH/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_HINDI_ENGLISH.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_hi_en_ner_pipeline_hi_3.4.1_3.0_1647954363761.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_hi_en_ner_pipeline_hi_3.4.1_3.0_1647954363761.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

## Results

```bash
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
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_hi_en_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
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