---
layout: model
title: NER Pipeline for German
author: John Snow Labs
name: xlm_roberta_large_token_classifier_conll03_pipeline
date: 2022-04-19
tags: [german, roberta, xlm, ner, conll03, de, open_source]
task: Named Entity Recognition
language: de
edition: Spark NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [xlm_roberta_large_token_classifier_conll03_de](https://nlp.johnsnowlabs.com/2021/12/25/xlm_roberta_large_token_classifier_conll03_de.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_conll03_pipeline_de_3.4.1_3.0_1650369924733.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_conll03_pipeline_de_3.4.1_3.0_1650369924733.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("xlm_roberta_large_token_classifier_conll03_pipeline", lang = "de")

pipeline.annotate("Ibser begann seine Karriere beim ASK Ebreichsdorf. 2004 wechselte er zu Admira Wacker Mödling, wo er auch in der Akademie spielte.")
```
```scala
val pipeline = new PretrainedPipeline("xlm_roberta_large_token_classifier_conll03_pipeline", lang = "de")

pipeline.annotate("Ibser begann seine Karriere beim ASK Ebreichsdorf. 2004 wechselte er zu Admira Wacker Mödling, wo er auch in der Akademie spielte.")
```
</div>

## Results

```bash
+----------------------+---------+
|chunk                 |ner_label|
+----------------------+---------+
|Ibser                 |PER      |
|ASK Ebreichsdorf      |ORG      |
|Admira Wacker Mödling |ORG      |
+----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_conll03_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 3.4.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|1.8 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- XlmRoBertaForTokenClassification
- NerConverter
- Finisher