---
layout: model
title: Dutch, Flemish roberta_qa_base_squad_pipeline pipeline RoBertaForQuestionAnswering from Nadav
author: John Snow Labs
name: roberta_qa_base_squad_pipeline
date: 2024-09-08
tags: [nl, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_qa_base_squad_pipeline` is a Dutch, Flemish model originally trained by Nadav.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_base_squad_pipeline_nl_5.5.0_3.0_1725833505945.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_qa_base_squad_pipeline_nl_5.5.0_3.0_1725833505945.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_qa_base_squad_pipeline", lang = "nl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_qa_base_squad_pipeline", lang = "nl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_qa_base_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|435.7 MB|

## References

https://huggingface.co/Nadav/roberta-base-squad-nl

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering