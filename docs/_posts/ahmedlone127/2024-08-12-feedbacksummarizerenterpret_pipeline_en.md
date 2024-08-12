---
layout: model
title: English feedbacksummarizerenterpret_pipeline pipeline T5Transformer from abhisheky127
author: John Snow Labs
name: feedbacksummarizerenterpret_pipeline
date: 2024-08-12
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`feedbacksummarizerenterpret_pipeline` is a English model originally trained by abhisheky127.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/feedbacksummarizerenterpret_pipeline_en_5.4.2_3.0_1723474976510.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/feedbacksummarizerenterpret_pipeline_en_5.4.2_3.0_1723474976510.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("feedbacksummarizerenterpret_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("feedbacksummarizerenterpret_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|feedbacksummarizerenterpret_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|316.7 MB|

## References

https://huggingface.co/abhisheky127/FeedbackSummarizerEnterpret

## Included Models

- DocumentAssembler
- T5Transformer