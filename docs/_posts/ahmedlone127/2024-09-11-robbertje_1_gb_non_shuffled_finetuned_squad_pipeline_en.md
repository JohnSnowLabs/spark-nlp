---
layout: model
title: English robbertje_1_gb_non_shuffled_finetuned_squad_pipeline pipeline RoBertaForQuestionAnswering from Ztijn
author: John Snow Labs
name: robbertje_1_gb_non_shuffled_finetuned_squad_pipeline
date: 2024-09-11
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`robbertje_1_gb_non_shuffled_finetuned_squad_pipeline` is a English model originally trained by Ztijn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/robbertje_1_gb_non_shuffled_finetuned_squad_pipeline_en_5.5.0_3.0_1726062340935.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/robbertje_1_gb_non_shuffled_finetuned_squad_pipeline_en_5.5.0_3.0_1726062340935.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("robbertje_1_gb_non_shuffled_finetuned_squad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("robbertje_1_gb_non_shuffled_finetuned_squad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|robbertje_1_gb_non_shuffled_finetuned_squad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|276.6 MB|

## References

https://huggingface.co/Ztijn/robbertje-1-gb-non-shuffled-finetuned-squad

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering