---
layout: model
title: English mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline pipeline XlmRoBertaEmbeddings from enod
author: John Snow Labs
name: mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline
date: 2024-09-01
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

Pretrained XlmRoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline` is a English model originally trained by enod.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline_en_5.4.2_3.0_1725177859382.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline_en_5.4.2_3.0_1725177859382.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mminilmv2_l6_h384_distilled_from_xlmr_large_enod_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|257.3 MB|

## References

https://huggingface.co/enod/mMiniLMv2-L6-H384-distilled-from-XLMR-Large

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaEmbeddings