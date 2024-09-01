---
layout: model
title: Italian umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline pipeline CamemBertForQuestionAnswering from mrm8488
author: John Snow Labs
name: umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline
date: 2024-09-01
tags: [it, open_source, pipeline, onnx]
task: Question Answering
language: it
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline` is a Italian model originally trained by mrm8488.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline_it_5.4.2_3.0_1725162872611.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline_it_5.4.2_3.0_1725162872611.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline", lang = "it")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline", lang = "it")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|umberto_wikipedia_uncased_v1_finetuned_squadv1_italian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|it|
|Size:|410.2 MB|

## References

https://huggingface.co/mrm8488/umberto-wikipedia-uncased-v1-finetuned-squadv1-it

## Included Models

- MultiDocumentAssembler
- CamemBertForQuestionAnswering