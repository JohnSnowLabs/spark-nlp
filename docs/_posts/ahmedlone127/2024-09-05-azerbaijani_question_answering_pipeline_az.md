---
layout: model
title: Azerbaijani azerbaijani_question_answering_pipeline pipeline RoBertaForQuestionAnswering from interneuronai
author: John Snow Labs
name: azerbaijani_question_answering_pipeline
date: 2024-09-05
tags: [az, open_source, pipeline, onnx]
task: Question Answering
language: az
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`azerbaijani_question_answering_pipeline` is a Azerbaijani model originally trained by interneuronai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/azerbaijani_question_answering_pipeline_az_5.5.0_3.0_1725576393936.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/azerbaijani_question_answering_pipeline_az_5.5.0_3.0_1725576393936.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("azerbaijani_question_answering_pipeline", lang = "az")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("azerbaijani_question_answering_pipeline", lang = "az")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|azerbaijani_question_answering_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|az|
|Size:|1.3 GB|

## References

https://huggingface.co/interneuronai/az-question-answering

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering