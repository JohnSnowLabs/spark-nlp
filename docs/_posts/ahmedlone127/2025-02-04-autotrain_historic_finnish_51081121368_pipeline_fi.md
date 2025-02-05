---
layout: model
title: Finnish autotrain_historic_finnish_51081121368_pipeline pipeline BertForTokenClassification from peanutacake
author: John Snow Labs
name: autotrain_historic_finnish_51081121368_pipeline
date: 2025-02-04
tags: [fi, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autotrain_historic_finnish_51081121368_pipeline` is a Finnish model originally trained by peanutacake.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autotrain_historic_finnish_51081121368_pipeline_fi_5.5.1_3.0_1738674200604.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autotrain_historic_finnish_51081121368_pipeline_fi_5.5.1_3.0_1738674200604.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("autotrain_historic_finnish_51081121368_pipeline", lang = "fi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("autotrain_historic_finnish_51081121368_pipeline", lang = "fi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|autotrain_historic_finnish_51081121368_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fi|
|Size:|464.7 MB|

## References

https://huggingface.co/peanutacake/autotrain-historic-fi-51081121368

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification