---
layout: model
title: English roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline pipeline RoBertaForQuestionAnswering from AnonymousSub
author: John Snow Labs
name: roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline
date: 2024-09-04
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline` is a English model originally trained by AnonymousSub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline_en_5.5.0_3.0_1725450777398.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline_en_5.5.0_3.0_1725450777398.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_qa_recipe_triplet_recipes_base_easy_timestep_squadv2_epochs_3_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|466.3 MB|

## References

https://huggingface.co/AnonymousSub/recipe_triplet_recipes-roberta-base_EASY_TIMESTEP_squadv2_epochs_3

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering