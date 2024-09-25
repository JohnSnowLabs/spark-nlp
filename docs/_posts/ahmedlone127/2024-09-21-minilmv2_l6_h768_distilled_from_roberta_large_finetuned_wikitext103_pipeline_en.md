---
layout: model
title: English minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline pipeline RoBertaEmbeddings from saghar
author: John Snow Labs
name: minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline
date: 2024-09-21
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline` is a English model originally trained by saghar.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline_en_5.5.0_3.0_1726943671538.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline_en_5.5.0_3.0_1726943671538.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|minilmv2_l6_h768_distilled_from_roberta_large_finetuned_wikitext103_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|196.0 MB|

## References

https://huggingface.co/saghar/MiniLMv2-L6-H768-distilled-from-RoBERTa-Large-finetuned-wikitext103

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings