---
layout: model
title: English xlmr_roberta_base_finetuned_panx_all_pipeline pipeline XlmRoBertaForTokenClassification from ibk007
author: John Snow Labs
name: xlmr_roberta_base_finetuned_panx_all_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmr_roberta_base_finetuned_panx_all_pipeline` is a English model originally trained by ibk007.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmr_roberta_base_finetuned_panx_all_pipeline_en_5.5.1_3.0_1743122735408.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmr_roberta_base_finetuned_panx_all_pipeline_en_5.5.1_3.0_1743122735408.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmr_roberta_base_finetuned_panx_all_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmr_roberta_base_finetuned_panx_all_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmr_roberta_base_finetuned_panx_all_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|849.0 MB|

## References

https://huggingface.co/ibk007/xlmr-roberta-base-finetuned-panx-all

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification