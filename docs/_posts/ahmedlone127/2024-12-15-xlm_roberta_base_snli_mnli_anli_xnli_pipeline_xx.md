---
layout: model
title: Multilingual xlm_roberta_base_snli_mnli_anli_xnli_pipeline pipeline XlmRoBertaForZeroShotClassification from symanto
author: John Snow Labs
name: xlm_roberta_base_snli_mnli_anli_xnli_pipeline
date: 2024-12-15
tags: [xx, open_source, pipeline, onnx]
task: Zero-Shot Classification
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForZeroShotClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_snli_mnli_anli_xnli_pipeline` is a Multilingual model originally trained by symanto.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_snli_mnli_anli_xnli_pipeline_xx_5.5.1_3.0_1734232917521.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_snli_mnli_anli_xnli_pipeline_xx_5.5.1_3.0_1734232917521.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_snli_mnli_anli_xnli_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_snli_mnli_anli_xnli_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_snli_mnli_anli_xnli_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|899.7 MB|

## References

https://huggingface.co/symanto/xlm-roberta-base-snli-mnli-anli-xnli

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForZeroShotClassification