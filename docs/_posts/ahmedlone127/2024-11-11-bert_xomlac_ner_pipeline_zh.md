---
layout: model
title: Chinese bert_xomlac_ner_pipeline pipeline BertForTokenClassification from b3x0m
author: John Snow Labs
name: bert_xomlac_ner_pipeline
date: 2024-11-11
tags: [zh, open_source, pipeline, onnx]
task: Named Entity Recognition
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bert_xomlac_ner_pipeline` is a Chinese model originally trained by b3x0m.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_xomlac_ner_pipeline_zh_5.5.1_3.0_1731298814714.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_xomlac_ner_pipeline_zh_5.5.1_3.0_1731298814714.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_xomlac_ner_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bert_xomlac_ner_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_xomlac_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|381.1 MB|

## References

https://huggingface.co/b3x0m/bert-xomlac-ner

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification