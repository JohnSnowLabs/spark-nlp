---
layout: model
title: English open_gpt_3_5_detector_pipeline pipeline DistilBertForSequenceClassification from nothingiisreal
author: John Snow Labs
name: open_gpt_3_5_detector_pipeline
date: 2024-09-08
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DistilBertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`open_gpt_3_5_detector_pipeline` is a English model originally trained by nothingiisreal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/open_gpt_3_5_detector_pipeline_en_5.5.0_3.0_1725808610371.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/open_gpt_3_5_detector_pipeline_en_5.5.0_3.0_1725808610371.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("open_gpt_3_5_detector_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("open_gpt_3_5_detector_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|open_gpt_3_5_detector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|249.5 MB|

## References

https://huggingface.co/nothingiisreal/open-gpt-3.5-detector

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForSequenceClassification