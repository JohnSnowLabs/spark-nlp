---
layout: model
title: English hs_arabic_translate_syn_4class_for_tool_pipeline pipeline BertForSequenceClassification from SoDehghan
author: John Snow Labs
name: hs_arabic_translate_syn_4class_for_tool_pipeline
date: 2024-11-11
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hs_arabic_translate_syn_4class_for_tool_pipeline` is a English model originally trained by SoDehghan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hs_arabic_translate_syn_4class_for_tool_pipeline_en_5.5.1_3.0_1731309392366.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hs_arabic_translate_syn_4class_for_tool_pipeline_en_5.5.1_3.0_1731309392366.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hs_arabic_translate_syn_4class_for_tool_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hs_arabic_translate_syn_4class_for_tool_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hs_arabic_translate_syn_4class_for_tool_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|507.5 MB|

## References

https://huggingface.co/SoDehghan/hs-ar-translate-syn-4class-for-tool

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification