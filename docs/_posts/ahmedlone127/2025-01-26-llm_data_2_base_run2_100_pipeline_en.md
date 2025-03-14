---
layout: model
title: English llm_data_2_base_run2_100_pipeline pipeline XlmRoBertaForSequenceClassification from sreddy109
author: John Snow Labs
name: llm_data_2_base_run2_100_pipeline
date: 2025-01-26
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`llm_data_2_base_run2_100_pipeline` is a English model originally trained by sreddy109.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/llm_data_2_base_run2_100_pipeline_en_5.5.1_3.0_1737886206804.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/llm_data_2_base_run2_100_pipeline_en_5.5.1_3.0_1737886206804.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("llm_data_2_base_run2_100_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("llm_data_2_base_run2_100_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|llm_data_2_base_run2_100_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|994.0 MB|

## References

https://huggingface.co/sreddy109/llm-data-2-base-run2-100

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification