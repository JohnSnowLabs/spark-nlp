---
layout: model
title: English recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline pipeline RoBertaForQuestionAnswering from AnonymousSub
author: John Snow Labs
name: recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline
date: 2024-09-15
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline` is a English model originally trained by AnonymousSub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline_en_5.5.0_3.0_1726369419030.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline_en_5.5.0_3.0_1726369419030.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|recipe_triplet_roberta_base_easy_squadv2_epochs_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|460.5 MB|

## References

https://huggingface.co/AnonymousSub/recipe_triplet_roberta-base_EASY_squadv2_epochs_3

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering