---
layout: model
title: NER Pipeline for 10 High Resourced Languages
author: John Snow Labs
name: xlm_roberta_large_token_classifier_hrl_pipeline
date: 2023-05-22
tags: [arabic, german, english, spanish, french, italian, latvian, dutch, portuguese, chinese, xlm, roberta, ner, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 4.4.2
spark_version: 3.2
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
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_hrl_pipeline_xx_4.4.2_3.2_1684761928074.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_hrl_pipeline_xx_4.4.2_3.2_1684761928074.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
Results



+---------------------------+---------+
|chunk                      |ner_label|
+---------------------------+---------+
|الرياض                     |LOC      |
|فيصل بن بندر بن عبد العزيز |PER      |
|الرياض                     |LOC      |
+---------------------------+---------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_hrl_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
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