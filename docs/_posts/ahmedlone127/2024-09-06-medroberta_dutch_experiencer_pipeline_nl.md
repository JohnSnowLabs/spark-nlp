---
layout: model
title: Dutch, Flemish medroberta_dutch_experiencer_pipeline pipeline RoBertaForTokenClassification from UMCU
author: John Snow Labs
name: medroberta_dutch_experiencer_pipeline
date: 2024-09-06
tags: [nl, open_source, pipeline, onnx]
task: Named Entity Recognition
language: nl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`medroberta_dutch_experiencer_pipeline` is a Dutch, Flemish model originally trained by UMCU.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/medroberta_dutch_experiencer_pipeline_nl_5.5.0_3.0_1725624655916.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/medroberta_dutch_experiencer_pipeline_nl_5.5.0_3.0_1725624655916.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("medroberta_dutch_experiencer_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("medroberta_dutch_experiencer_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|medroberta_dutch_experiencer_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|469.8 MB|

## References

https://huggingface.co/UMCU/MedRoBERTa.nl_Experiencer

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification