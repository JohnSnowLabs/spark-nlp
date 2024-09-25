---
layout: model
title: English gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline pipeline BertForSequenceClassification from tanoManzo
author: John Snow Labs
name: gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline
date: 2024-09-18
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

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline` is a English model originally trained by tanoManzo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline_en_5.5.0_3.0_1726647930756.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline_en_5.5.0_3.0_1726647930756.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gena_lm_bert_base_t2t_ft_hepg2_1kbphg19_dhss_h3k27ac_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|414.0 MB|

## References

https://huggingface.co/tanoManzo/gena-lm-bert-base-t2t_ft_Hepg2_1kbpHG19_DHSs_H3K27AC

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification