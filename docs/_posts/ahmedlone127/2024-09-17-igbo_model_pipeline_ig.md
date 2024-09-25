---
layout: model
title: Igbo igbo_model_pipeline pipeline XlmRoBertaForTokenClassification from ignatius
author: John Snow Labs
name: igbo_model_pipeline
date: 2024-09-17
tags: [ig, open_source, pipeline, onnx]
task: Named Entity Recognition
language: ig
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`igbo_model_pipeline` is a Igbo model originally trained by ignatius.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/igbo_model_pipeline_ig_5.5.0_3.0_1726577158736.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/igbo_model_pipeline_ig_5.5.0_3.0_1726577158736.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("igbo_model_pipeline", lang = "ig")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("igbo_model_pipeline", lang = "ig")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|igbo_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ig|
|Size:|443.2 MB|

## References

https://huggingface.co/ignatius/igbo_model

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification