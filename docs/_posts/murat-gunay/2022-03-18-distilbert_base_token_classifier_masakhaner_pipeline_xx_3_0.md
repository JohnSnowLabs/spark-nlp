---
layout: model
title: NER Pipeline for 9 African Languages
author: John Snow Labs
name: distilbert_base_token_classifier_masakhaner_pipeline
date: 2022-03-18
tags: [hausa, igbo, kinyarwanda, luganda, nigerian, pidgin, swahilu, wolof, yoruba, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [distilbert_base_token_classifier_masakhaner](https://nlp.johnsnowlabs.com/2022/01/18/distilbert_base_token_classifier_masakhaner_xx.html) model.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/Ner_masakhaner/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Ner_masakhaner.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_token_classifier_masakhaner_pipeline_xx_3.4.1_3.0_1647607984667.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
masakhaner_pipeline = PretrainedPipeline("distilbert_base_token_classifier_masakhaner_pipeline", lang = "xx")

masakhaner_pipeline.annotate("Ilé-iṣẹ́ẹ Mohammed Sani Musa, Activate Technologies Limited, ni ó kó ẹ̀rọ Ìwé-pélébé Ìdìbò Alálòpẹ́ (PVCs) tí a lò fún ìbò ọdún-un 2019, ígbà tí ó jẹ́ òǹdíjedupò lábẹ́ ẹgbẹ́ olóṣèlúu tí ó ń tukọ̀ ètò ìṣèlú lọ́wọ́ All rogressives Congress (APC) fún Aṣojú Ìlà-Oòrùn Niger, ìyẹn gẹ́gẹ́ bí ilé iṣẹ́ aṣèwádìí, Premium Times ṣe tẹ̀ ẹ́ jáde.")
```
```scala
val masakhaner_pipeline = new PretrainedPipeline("distilbert_base_token_classifier_masakhaner_pipeline", lang = "xx")

masakhaner_pipeline.annotate("Ilé-iṣẹ́ẹ Mohammed Sani Musa, Activate Technologies Limited, ni ó kó ẹ̀rọ Ìwé-pélébé Ìdìbò Alálòpẹ́ (PVCs) tí a lò fún ìbò ọdún-un 2019, ígbà tí ó jẹ́ òǹdíjedupò lábẹ́ ẹgbẹ́ olóṣèlúu tí ó ń tukọ̀ ètò ìṣèlú lọ́wọ́ All rogressives Congress (APC) fún Aṣojú Ìlà-Oòrùn Niger, ìyẹn gẹ́gẹ́ bí ilé iṣẹ́ aṣèwádìí, Premium Times ṣe tẹ̀ ẹ́ jáde.")
```
</div>

## Results

```bash
+-----------------------------+---------+
|chunk                        |ner_label|
+-----------------------------+---------+
|Mohammed Sani Musa           |PER      |
|Activate Technologies Limited|ORG      |
|ọdún-un 2019                 |DATE     |
|All rogressives Congress     |ORG      |
|APC                          |ORG      |
|Aṣojú Ìlà-Oòrùn Niger        |LOC      |
|Premium Times                |ORG      |
+-----------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_token_classifier_masakhaner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|505.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- DistilBertForTokenClassification
- NerConverter
- Finisher