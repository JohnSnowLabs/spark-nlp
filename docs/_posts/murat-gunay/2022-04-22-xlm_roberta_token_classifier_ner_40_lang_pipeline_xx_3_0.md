---
layout: model
title: XLM-RoBERTa 40-Language NER Pipeline
author: John Snow Labs
name: xlm_roberta_token_classifier_ner_40_lang_pipeline
date: 2022-04-22
tags: [open_source, ner, token_classifier, xlm_roberta, multilang, "40", xx]
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

This pretrained pipeline is built on the top of [xlm_roberta_token_classifier_ner_40_lang](https://nlp.johnsnowlabs.com/2021/09/28/xlm_roberta_token_classifier_ner_40_lang_xx.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_token_classifier_ner_40_lang_pipeline_xx_3.4.1_3.0_1650628752833.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_token_classifier_ner_40_lang_pipeline_xx_3.4.1_3.0_1650628752833.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("xlm_roberta_token_classifier_ner_40_lang_pipeline", lang = "xx")

pipeline.annotate(["My name is John and I work at John Snow Labs.", "انا اسمي احمد واعمل في ارامكو"])
```
```scala
val pipeline = new PretrainedPipeline("xlm_roberta_token_classifier_ner_40_lang_pipeline", lang = "xx")

pipeline.annotate(Array("My name is John and I work at John Snow Labs.", "انا اسمي احمد واعمل في ارامكو"))
```
</div>

## Results

```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|John          |PER      |
|John Snow Labs|ORG      |
|احمد          |PER      |
|ارامكو        |ORG      |
+--------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_token_classifier_ner_40_lang_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|967.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- XlmRoBertaForTokenClassification
- NerConverter
- Finisher