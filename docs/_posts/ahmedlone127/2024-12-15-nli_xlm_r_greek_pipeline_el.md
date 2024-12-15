---
layout: model
title: Modern Greek (1453-) nli_xlm_r_greek_pipeline pipeline XlmRoBertaForZeroShotClassification from lighteternal
author: John Snow Labs
name: nli_xlm_r_greek_pipeline
date: 2024-12-15
tags: [el, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: el
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nli_xlm_r_greek_pipeline` is a Modern Greek (1453-) model originally trained by lighteternal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nli_xlm_r_greek_pipeline_el_5.5.1_3.0_1734232396853.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nli_xlm_r_greek_pipeline_el_5.5.1_3.0_1734232396853.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nli_xlm_r_greek_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nli_xlm_r_greek_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nli_xlm_r_greek_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|878.8 MB|

## References

https://huggingface.co/lighteternal/nli-xlm-r-greek

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForZeroShotClassification