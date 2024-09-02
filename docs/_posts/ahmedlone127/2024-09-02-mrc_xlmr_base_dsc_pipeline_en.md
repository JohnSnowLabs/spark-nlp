---
layout: model
title: English mrc_xlmr_base_dsc_pipeline pipeline XlmRoBertaForQuestionAnswering from MiuN2k3
author: John Snow Labs
name: mrc_xlmr_base_dsc_pipeline
date: 2024-09-02
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mrc_xlmr_base_dsc_pipeline` is a English model originally trained by MiuN2k3.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mrc_xlmr_base_dsc_pipeline_en_5.5.0_3.0_1725254114807.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mrc_xlmr_base_dsc_pipeline_en_5.5.0_3.0_1725254114807.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mrc_xlmr_base_dsc_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mrc_xlmr_base_dsc_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mrc_xlmr_base_dsc_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|815.8 MB|

## References

https://huggingface.co/MiuN2k3/mrc-xlmr-base-dsc

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering