---
layout: model
title: Icelandic declension_error_detection_pipeline pipeline RoBertaForSequenceClassification from IsakG
author: John Snow Labs
name: declension_error_detection_pipeline
date: 2025-04-05
tags: [is, open_source, pipeline, onnx]
task: Text Classification
language: is
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`declension_error_detection_pipeline` is a Icelandic model originally trained by IsakG.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/declension_error_detection_pipeline_is_5.5.1_3.0_1743827917527.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/declension_error_detection_pipeline_is_5.5.1_3.0_1743827917527.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("declension_error_detection_pipeline", lang = "is")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("declension_error_detection_pipeline", lang = "is")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|declension_error_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|463.3 MB|

## References

https://huggingface.co/IsakG/declension_error_detection

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification