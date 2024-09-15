---
layout: model
title: English output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline pipeline RoBertaForQuestionAnswering from AnonymousSub
author: John Snow Labs
name: output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline
date: 2024-09-12
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline` is a English model originally trained by AnonymousSub.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline_en_5.5.0_3.0_1726175552936.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline_en_5.5.0_3.0_1726175552936.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_large_epochs_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/AnonymousSub/output_mask_step_pretraining_plus_contr_roberta_model_from_pretrained_LARGE_EPOCHS_1

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering