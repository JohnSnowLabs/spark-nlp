---
layout: model
title: Portuguese portuguese_capitalization_punctuation_restoration_pipeline pipeline XlmRoBertaForTokenClassification from UMUTeam
author: John Snow Labs
name: portuguese_capitalization_punctuation_restoration_pipeline
date: 2024-09-04
tags: [pt, open_source, pipeline, onnx]
task: Named Entity Recognition
language: pt
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`portuguese_capitalization_punctuation_restoration_pipeline` is a Portuguese model originally trained by UMUTeam.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/portuguese_capitalization_punctuation_restoration_pipeline_pt_5.5.0_3.0_1725424035358.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/portuguese_capitalization_punctuation_restoration_pipeline_pt_5.5.0_3.0_1725424035358.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("portuguese_capitalization_punctuation_restoration_pipeline", lang = "pt")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("portuguese_capitalization_punctuation_restoration_pipeline", lang = "pt")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|portuguese_capitalization_punctuation_restoration_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|pt|
|Size:|883.3 MB|

## References

https://huggingface.co/UMUTeam/portuguese_capitalization_punctuation_restoration

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification