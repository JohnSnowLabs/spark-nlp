---
layout: model
title: Dutch, Flemish mdebertav3_subjectivity_dutch_pipeline pipeline DeBertaForSequenceClassification from GroNLP
author: John Snow Labs
name: mdebertav3_subjectivity_dutch_pipeline
date: 2024-09-09
tags: [nl, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdebertav3_subjectivity_dutch_pipeline` is a Dutch, Flemish model originally trained by GroNLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdebertav3_subjectivity_dutch_pipeline_nl_5.5.0_3.0_1725879027724.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdebertav3_subjectivity_dutch_pipeline_nl_5.5.0_3.0_1725879027724.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdebertav3_subjectivity_dutch_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdebertav3_subjectivity_dutch_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdebertav3_subjectivity_dutch_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|859.5 MB|

## References

https://huggingface.co/GroNLP/mdebertav3-subjectivity-dutch

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification