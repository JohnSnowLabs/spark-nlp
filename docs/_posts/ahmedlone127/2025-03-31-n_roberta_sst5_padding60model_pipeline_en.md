---
layout: model
title: English n_roberta_sst5_padding60model_pipeline pipeline RoBertaForSequenceClassification from Realgon
author: John Snow Labs
name: n_roberta_sst5_padding60model_pipeline
date: 2025-03-31
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`n_roberta_sst5_padding60model_pipeline` is a English model originally trained by Realgon.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/n_roberta_sst5_padding60model_pipeline_en_5.5.1_3.0_1743439319255.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/n_roberta_sst5_padding60model_pipeline_en_5.5.1_3.0_1743439319255.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("n_roberta_sst5_padding60model_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("n_roberta_sst5_padding60model_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|n_roberta_sst5_padding60model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|439.3 MB|

## References

https://huggingface.co/Realgon/N_roberta_sst5_padding60model

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification