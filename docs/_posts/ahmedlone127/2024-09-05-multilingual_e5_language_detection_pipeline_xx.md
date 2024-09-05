---
layout: model
title: Multilingual multilingual_e5_language_detection_pipeline pipeline XlmRoBertaForSequenceClassification from Mike0307
author: John Snow Labs
name: multilingual_e5_language_detection_pipeline
date: 2024-09-05
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_e5_language_detection_pipeline` is a Multilingual model originally trained by Mike0307.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_e5_language_detection_pipeline_xx_5.5.0_3.0_1725515245765.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_e5_language_detection_pipeline_xx_5.5.0_3.0_1725515245765.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("multilingual_e5_language_detection_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("multilingual_e5_language_detection_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|multilingual_e5_language_detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|887.2 MB|

## References

https://huggingface.co/Mike0307/multilingual-e5-language-detection

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification