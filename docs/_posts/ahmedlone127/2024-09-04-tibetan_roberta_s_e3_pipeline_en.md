---
layout: model
title: English tibetan_roberta_s_e3_pipeline pipeline RoBertaEmbeddings from spsither
author: John Snow Labs
name: tibetan_roberta_s_e3_pipeline
date: 2024-09-04
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tibetan_roberta_s_e3_pipeline` is a English model originally trained by spsither.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tibetan_roberta_s_e3_pipeline_en_5.5.0_3.0_1725412184438.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tibetan_roberta_s_e3_pipeline_en_5.5.0_3.0_1725412184438.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tibetan_roberta_s_e3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tibetan_roberta_s_e3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tibetan_roberta_s_e3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|311.4 MB|

## References

https://huggingface.co/spsither/tibetan_RoBERTa_S_e3

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings