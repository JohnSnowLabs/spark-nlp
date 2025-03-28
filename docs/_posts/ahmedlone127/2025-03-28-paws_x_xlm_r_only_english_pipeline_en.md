---
layout: model
title: English paws_x_xlm_r_only_english_pipeline pipeline XlmRoBertaForSequenceClassification from semindan
author: John Snow Labs
name: paws_x_xlm_r_only_english_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`paws_x_xlm_r_only_english_pipeline` is a English model originally trained by semindan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/paws_x_xlm_r_only_english_pipeline_en_5.5.1_3.0_1743188148530.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/paws_x_xlm_r_only_english_pipeline_en_5.5.1_3.0_1743188148530.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("paws_x_xlm_r_only_english_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("paws_x_xlm_r_only_english_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|paws_x_xlm_r_only_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|800.5 MB|

## References

https://huggingface.co/semindan/paws_x_xlm_r_only_en

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification