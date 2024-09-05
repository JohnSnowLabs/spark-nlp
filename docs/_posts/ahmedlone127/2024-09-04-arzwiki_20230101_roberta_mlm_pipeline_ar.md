---
layout: model
title: Arabic arzwiki_20230101_roberta_mlm_pipeline pipeline RoBertaEmbeddings from SaiedAlshahrani
author: John Snow Labs
name: arzwiki_20230101_roberta_mlm_pipeline
date: 2024-09-04
tags: [ar, open_source, pipeline, onnx]
task: Embeddings
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arzwiki_20230101_roberta_mlm_pipeline` is a Arabic model originally trained by SaiedAlshahrani.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arzwiki_20230101_roberta_mlm_pipeline_ar_5.5.0_3.0_1725412435549.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arzwiki_20230101_roberta_mlm_pipeline_ar_5.5.0_3.0_1725412435549.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arzwiki_20230101_roberta_mlm_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arzwiki_20230101_roberta_mlm_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arzwiki_20230101_roberta_mlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|311.8 MB|

## References

https://huggingface.co/SaiedAlshahrani/arzwiki_20230101_roberta_mlm

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings