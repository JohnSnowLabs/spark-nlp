---
layout: model
title: NER Pipeline for 10 High Resourced Languages
author: John Snow Labs
name: xlm_roberta_large_token_classifier_hrl_pipeline
date: 2022-06-27
tags: [arabic, german, english, spanish, french, italian, latvian, dutch, portuguese, chinese, xlm, roberta, ner, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [xlm_roberta_large_token_classifier_hrl](https://nlp.johnsnowlabs.com/2021/12/26/xlm_roberta_large_token_classifier_hrl_xx.html) model.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_HRL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/Ner_HRL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_hrl_pipeline_xx_4.0.0_3.0_1656371823877.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_hrl_pipeline_xx_4.0.0_3.0_1656371823877.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_large_token_classifier_hrl_pipeline", lang = "xx")

pipeline.annotate("يمكنكم مشاهدة أمير منطقة الرياض الأمير فيصل بن بندر بن عبد العزيز في كل مناسبة وافتتاح تتعلق بمشاريع التعليم والصحة وخدمة الطرق والمشاريع الثقافية في منطقة الرياض.")
```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_large_token_classifier_hrl_pipeline", lang = "xx")

pipeline.annotate("يمكنكم مشاهدة أمير منطقة الرياض الأمير فيصل بن بندر بن عبد العزيز في كل مناسبة وافتتاح تتعلق بمشاريع التعليم والصحة وخدمة الطرق والمشاريع الثقافية في منطقة الرياض.")
```
</div>

## Results

```bash

+---------------------------+---------+
|chunk                      |ner_label|
+---------------------------+---------+
|الرياض                     |LOC      |
|فيصل بن بندر بن عبد العزيز |PER      |
|الرياض                     |LOC      |
+---------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_hrl_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.8 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- XlmRoBertaForTokenClassification
- NerConverter
- Finisher