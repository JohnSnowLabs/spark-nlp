---
layout: model
title: English autotrain_ajmc_english_ner_52850124468_pipeline pipeline BertForTokenClassification from peanutacake
author: John Snow Labs
name: autotrain_ajmc_english_ner_52850124468_pipeline
date: 2025-02-02
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_ajmc_english_ner_52850124468_pipeline` is a English model originally trained by peanutacake.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_ajmc_english_ner_52850124468_pipeline_en_5.5.1_3.0_1738470243217.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_ajmc_english_ner_52850124468_pipeline_en_5.5.1_3.0_1738470243217.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_ajmc_english_ner_52850124468_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_ajmc_english_ner_52850124468_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_ajmc_english_ner_52850124468_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.2 GB|

## References

https://huggingface.co/peanutacake/autotrain-ajmc_en_ner-52850124468

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification