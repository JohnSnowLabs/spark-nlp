---
layout: model
title: Amharic ft_exlmr_pipeline pipeline XlmRoBertaForSequenceClassification from Hailay
author: John Snow Labs
name: ft_exlmr_pipeline
date: 2024-12-15
tags: [am, open_source, pipeline, onnx]
task: Text Classification
language: am
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ft_exlmr_pipeline` is a Amharic model originally trained by Hailay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ft_exlmr_pipeline_am_5.5.1_3.0_1734292783815.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ft_exlmr_pipeline_am_5.5.1_3.0_1734292783815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ft_exlmr_pipeline", lang = "am")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ft_exlmr_pipeline", lang = "am")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ft_exlmr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|am|
|Size:|907.8 MB|

## References

https://huggingface.co/Hailay/FT_EXLMR

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification