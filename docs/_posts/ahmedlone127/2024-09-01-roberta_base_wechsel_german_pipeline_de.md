---
layout: model
title: German roberta_base_wechsel_german_pipeline pipeline RoBertaEmbeddings from benjamin
author: John Snow Labs
name: roberta_base_wechsel_german_pipeline
date: 2024-09-01
tags: [de, open_source, pipeline, onnx]
task: Embeddings
language: de
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_wechsel_german_pipeline` is a German model originally trained by benjamin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_german_pipeline_de_5.4.2_3.0_1725165291342.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_wechsel_german_pipeline_de_5.4.2_3.0_1725165291342.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_wechsel_german_pipeline", lang = "de")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_wechsel_german_pipeline", lang = "de")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_wechsel_german_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|de|
|Size:|465.8 MB|

## References

https://huggingface.co/benjamin/roberta-base-wechsel-german

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings