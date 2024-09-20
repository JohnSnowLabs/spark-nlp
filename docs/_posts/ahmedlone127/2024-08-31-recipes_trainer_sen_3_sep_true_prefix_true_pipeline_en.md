---
layout: model
title: English recipes_trainer_sen_3_sep_true_prefix_true_pipeline pipeline CamemBertEmbeddings from comartinez
author: John Snow Labs
name: recipes_trainer_sen_3_sep_true_prefix_true_pipeline
date: 2024-08-31
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`recipes_trainer_sen_3_sep_true_prefix_true_pipeline` is a English model originally trained by comartinez.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/recipes_trainer_sen_3_sep_true_prefix_true_pipeline_en_5.4.2_3.0_1725135940465.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/recipes_trainer_sen_3_sep_true_prefix_true_pipeline_en_5.4.2_3.0_1725135940465.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("recipes_trainer_sen_3_sep_true_prefix_true_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("recipes_trainer_sen_3_sep_true_prefix_true_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|recipes_trainer_sen_3_sep_true_prefix_true_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|412.4 MB|

## References

https://huggingface.co/comartinez/recipes-trainer_sen_3_sep_True_prefix_True

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertEmbeddings