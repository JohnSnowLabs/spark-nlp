---
layout: model
title: English finetuned_roberta_base_squad2_model_for_insurance_data_pipeline pipeline RoBertaForQuestionAnswering from sprateek
author: John Snow Labs
name: finetuned_roberta_base_squad2_model_for_insurance_data_pipeline
date: 2024-09-03
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

Pretrained RoBertaForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_roberta_base_squad2_model_for_insurance_data_pipeline` is a English model originally trained by sprateek.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_roberta_base_squad2_model_for_insurance_data_pipeline_en_5.5.0_3.0_1725370334453.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_roberta_base_squad2_model_for_insurance_data_pipeline_en_5.5.0_3.0_1725370334453.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_roberta_base_squad2_model_for_insurance_data_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_roberta_base_squad2_model_for_insurance_data_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_roberta_base_squad2_model_for_insurance_data_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|463.6 MB|

## References

https://huggingface.co/sprateek/finetuned-roberta-base-squad2-model-for-insurance-data

## Included Models

- MultiDocumentAssembler
- RoBertaForQuestionAnswering