---
layout: model
title: English vila_roberta_large_s2vl_internal_pipeline pipeline RoBertaForTokenClassification from allenai
author: John Snow Labs
name: vila_roberta_large_s2vl_internal_pipeline
date: 2024-09-02
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`vila_roberta_large_s2vl_internal_pipeline` is a English model originally trained by allenai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/vila_roberta_large_s2vl_internal_pipeline_en_5.5.0_3.0_1725310912419.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/vila_roberta_large_s2vl_internal_pipeline_en_5.5.0_3.0_1725310912419.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("vila_roberta_large_s2vl_internal_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("vila_roberta_large_s2vl_internal_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|vila_roberta_large_s2vl_internal_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/allenai/vila-roberta-large-s2vl-internal

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification