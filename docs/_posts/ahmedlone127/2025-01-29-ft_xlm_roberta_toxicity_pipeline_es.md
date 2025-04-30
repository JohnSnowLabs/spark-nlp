---
layout: model
title: Castilian, Spanish ft_xlm_roberta_toxicity_pipeline pipeline XlmRoBertaForSequenceClassification from bgonzalezbustamante
author: John Snow Labs
name: ft_xlm_roberta_toxicity_pipeline
date: 2025-01-29
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ft_xlm_roberta_toxicity_pipeline` is a Castilian, Spanish model originally trained by bgonzalezbustamante.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ft_xlm_roberta_toxicity_pipeline_es_5.5.1_3.0_1738177944897.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ft_xlm_roberta_toxicity_pipeline_es_5.5.1_3.0_1738177944897.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ft_xlm_roberta_toxicity_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ft_xlm_roberta_toxicity_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ft_xlm_roberta_toxicity_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|779.0 MB|

## References

https://huggingface.co/bgonzalezbustamante/ft-xlm-roberta-toxicity

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification