---
layout: model
title: English topic_modelling_roberta_pipeline pipeline RoBertaForSequenceClassification from ridhodaffasyah
author: John Snow Labs
name: topic_modelling_roberta_pipeline
date: 2024-09-13
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`topic_modelling_roberta_pipeline` is a English model originally trained by ridhodaffasyah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/topic_modelling_roberta_pipeline_en_5.5.0_3.0_1726246850996.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/topic_modelling_roberta_pipeline_en_5.5.0_3.0_1726246850996.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("topic_modelling_roberta_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("topic_modelling_roberta_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|topic_modelling_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|468.4 MB|

## References

https://huggingface.co/ridhodaffasyah/topic-modelling-roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification