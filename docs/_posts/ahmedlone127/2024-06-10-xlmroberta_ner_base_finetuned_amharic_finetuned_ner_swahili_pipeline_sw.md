---
layout: model
title: Swahili (macrolanguage) xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline pipeline XlmRoBertaForTokenClassification from mbeukman
author: John Snow Labs
name: xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline
date: 2024-06-10
tags: [sw, open_source, pipeline, onnx]
task: Named Entity Recognition
language: sw
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline` is a Swahili (macrolanguage) model originally trained by mbeukman.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline_sw_5.4.0_3.0_1718010718548.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline_sw_5.4.0_3.0_1718010718548.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline", lang = "sw")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline", lang = "sw")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_ner_base_finetuned_amharic_finetuned_ner_swahili_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|sw|
|Size:|1.0 GB|

## References

https://huggingface.co/mbeukman/xlm-roberta-base-finetuned-amharic-finetuned-ner-swahili

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification