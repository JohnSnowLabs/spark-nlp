---
layout: model
title: Dzongkha dzoqa_tibet_pipeline pipeline RoBertaForQuestionAnswering from Norphel
author: John Snow Labs
name: dzoqa_tibet_pipeline
date: 2024-09-13
tags: [dz, open_source, pipeline, onnx]
task: Question Answering
language: dz
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dzoqa_tibet_pipeline` is a Dzongkha model originally trained by Norphel.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dzoqa_tibet_pipeline_dz_5.5.0_3.0_1726206888503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dzoqa_tibet_pipeline_dz_5.5.0_3.0_1726206888503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dzoqa_tibet_pipeline", lang = "dz")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dzoqa_tibet_pipeline", lang = "dz")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dzoqa_tibet_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|dz|
|Size:|311.2 MB|

## References

https://huggingface.co/Norphel/dzoQA_tibet

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering