---
layout: model
title: Castilian, Spanish sentimientos_bullrich_pipeline pipeline XlmRoBertaForSequenceClassification from natmarinn
author: John Snow Labs
name: sentimientos_bullrich_pipeline
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sentimientos_bullrich_pipeline` is a Castilian, Spanish model originally trained by natmarinn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sentimientos_bullrich_pipeline_es_5.5.1_3.0_1738126999323.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sentimientos_bullrich_pipeline_es_5.5.1_3.0_1738126999323.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sentimientos_bullrich_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sentimientos_bullrich_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sentimientos_bullrich_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|1.0 GB|

## References

https://huggingface.co/natmarinn/sentimientos-bullrich

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification