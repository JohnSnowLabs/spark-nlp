---
layout: model
title: Hebrew golempii_v1_pipeline pipeline XlmRoBertaForTokenClassification from CordwainerSmith
author: John Snow Labs
name: golempii_v1_pipeline
date: 2025-01-25
tags: [he, open_source, pipeline, onnx]
task: Named Entity Recognition
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`golempii_v1_pipeline` is a Hebrew model originally trained by CordwainerSmith.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/golempii_v1_pipeline_he_5.5.1_3.0_1737792361279.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/golempii_v1_pipeline_he_5.5.1_3.0_1737792361279.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("golempii_v1_pipeline", lang = "he")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("golempii_v1_pipeline", lang = "he")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|golempii_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|he|
|Size:|833.8 MB|

## References

https://huggingface.co/CordwainerSmith/GolemPII-v1

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification