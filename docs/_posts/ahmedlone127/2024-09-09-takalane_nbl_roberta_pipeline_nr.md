---
layout: model
title: South Ndebele takalane_nbl_roberta_pipeline pipeline RoBertaEmbeddings from jannesg
author: John Snow Labs
name: takalane_nbl_roberta_pipeline
date: 2024-09-09
tags: [nr, open_source, pipeline, onnx]
task: Embeddings
language: nr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`takalane_nbl_roberta_pipeline` is a South Ndebele model originally trained by jannesg.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/takalane_nbl_roberta_pipeline_nr_5.5.0_3.0_1725925626484.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/takalane_nbl_roberta_pipeline_nr_5.5.0_3.0_1725925626484.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("takalane_nbl_roberta_pipeline", lang = "nr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("takalane_nbl_roberta_pipeline", lang = "nr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|takalane_nbl_roberta_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nr|
|Size:|310.2 MB|

## References

https://huggingface.co/jannesg/takalane_nbl_roberta

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings