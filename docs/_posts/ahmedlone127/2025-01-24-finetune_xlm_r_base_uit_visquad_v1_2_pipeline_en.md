---
layout: model
title: English finetune_xlm_r_base_uit_visquad_v1_2_pipeline pipeline XlmRoBertaForQuestionAnswering from haidangnguyen467
author: John Snow Labs
name: finetune_xlm_r_base_uit_visquad_v1_2_pipeline
date: 2025-01-24
tags: [en, open_source, pipeline, onnx]
task: Question Answering
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetune_xlm_r_base_uit_visquad_v1_2_pipeline` is a English model originally trained by haidangnguyen467.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetune_xlm_r_base_uit_visquad_v1_2_pipeline_en_5.5.1_3.0_1737753503830.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetune_xlm_r_base_uit_visquad_v1_2_pipeline_en_5.5.1_3.0_1737753503830.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetune_xlm_r_base_uit_visquad_v1_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetune_xlm_r_base_uit_visquad_v1_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetune_xlm_r_base_uit_visquad_v1_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|814.8 MB|

## References

https://huggingface.co/haidangnguyen467/finetune-xlm-r-base-uit-visquad-v1_2

## Included Models

- MultiDocumentAssembler
- XlmRoBertaForQuestionAnswering