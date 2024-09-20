---
layout: model
title: Modern Greek (1453-) bertatweetgr_pipeline pipeline RoBertaEmbeddings from Konstantinos
author: John Snow Labs
name: bertatweetgr_pipeline
date: 2024-09-13
tags: [el, open_source, pipeline, onnx]
task: Embeddings
language: el
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bertatweetgr_pipeline` is a Modern Greek (1453-) model originally trained by Konstantinos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bertatweetgr_pipeline_el_5.5.0_3.0_1726197597591.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bertatweetgr_pipeline_el_5.5.0_3.0_1726197597591.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bertatweetgr_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bertatweetgr_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bertatweetgr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|312.0 MB|

## References

https://huggingface.co/Konstantinos/BERTaTweetGR

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings