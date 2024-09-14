---
layout: model
title: Castilian, Spanish ironial52_roberta_pipeline pipeline RoBertaForSequenceClassification from l52mas
author: John Snow Labs
name: ironial52_roberta_pipeline
date: 2024-09-11
tags: [es, open_source, pipeline, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ironial52_roberta_pipeline` is a Castilian, Spanish model originally trained by l52mas.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ironial52_roberta_pipeline_es_5.5.0_3.0_1726082608661.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ironial52_roberta_pipeline_es_5.5.0_3.0_1726082608661.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ironial52_roberta_pipeline", lang = "es")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ironial52_roberta_pipeline", lang = "es")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ironial52_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|468.2 MB|

## References

https://huggingface.co/l52mas/ironiaL52_roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification