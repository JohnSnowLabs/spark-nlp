---
layout: model
title: Vietnamese tech_roberta_pipeline pipeline XlmRoBertaEmbeddings from imta-ai
author: John Snow Labs
name: tech_roberta_pipeline
date: 2024-09-05
tags: [vi, open_source, pipeline, onnx]
task: Embeddings
language: vi
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tech_roberta_pipeline` is a Vietnamese model originally trained by imta-ai.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tech_roberta_pipeline_vi_5.5.0_3.0_1725555933413.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tech_roberta_pipeline_vi_5.5.0_3.0_1725555933413.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tech_roberta_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tech_roberta_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tech_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|942.9 MB|

## References

https://huggingface.co/imta-ai/tech-roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings