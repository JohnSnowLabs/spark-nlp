---
layout: model
title: English edos_2023_baseline_xlm_roberta_base_label_category_pipeline pipeline XlmRoBertaForSequenceClassification from lct-rug-2022
author: John Snow Labs
name: edos_2023_baseline_xlm_roberta_base_label_category_pipeline
date: 2024-09-17
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`edos_2023_baseline_xlm_roberta_base_label_category_pipeline` is a English model originally trained by lct-rug-2022.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/edos_2023_baseline_xlm_roberta_base_label_category_pipeline_en_5.5.0_3.0_1726616673847.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/edos_2023_baseline_xlm_roberta_base_label_category_pipeline_en_5.5.0_3.0_1726616673847.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("edos_2023_baseline_xlm_roberta_base_label_category_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("edos_2023_baseline_xlm_roberta_base_label_category_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|edos_2023_baseline_xlm_roberta_base_label_category_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|654.5 MB|

## References

https://huggingface.co/lct-rug-2022/edos-2023-baseline-xlm-roberta-base-label_category

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification