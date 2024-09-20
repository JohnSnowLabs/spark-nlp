---
layout: model
title: English keyword_tag_model_4000_9_16_pipeline pipeline BertForTokenClassification from Media1129
author: John Snow Labs
name: keyword_tag_model_4000_9_16_pipeline
date: 2024-09-18
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`keyword_tag_model_4000_9_16_pipeline` is a English model originally trained by Media1129.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/keyword_tag_model_4000_9_16_pipeline_en_5.5.0_3.0_1726679278206.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/keyword_tag_model_4000_9_16_pipeline_en_5.5.0_3.0_1726679278206.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("keyword_tag_model_4000_9_16_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("keyword_tag_model_4000_9_16_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|keyword_tag_model_4000_9_16_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.2 MB|

## References

https://huggingface.co/Media1129/keyword-tag-model-4000-9-16

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification