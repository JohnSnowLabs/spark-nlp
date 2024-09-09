---
layout: model
title: Marathi roberta_mlm_marathi_pipeline pipeline RoBertaEmbeddings from deepampatel
author: John Snow Labs
name: roberta_mlm_marathi_pipeline
date: 2024-09-09
tags: [mr, open_source, pipeline, onnx]
task: Embeddings
language: mr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_mlm_marathi_pipeline` is a Marathi model originally trained by deepampatel.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_mlm_marathi_pipeline_mr_5.5.0_3.0_1725860255569.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_mlm_marathi_pipeline_mr_5.5.0_3.0_1725860255569.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_mlm_marathi_pipeline", lang = "mr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_mlm_marathi_pipeline", lang = "mr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_mlm_marathi_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|mr|
|Size:|470.9 MB|

## References

https://huggingface.co/deepampatel/roberta-mlm-marathi

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings