---
layout: model
title: English re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline pipeline XlmRoBertaForTokenClassification from ajtamayoh
author: John Snow Labs
name: re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline
date: 2024-09-02
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline` is a English model originally trained by ajtamayoh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline_en_5.5.0_3.0_1725308312312.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline_en_5.5.0_3.0_1725308312312.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_negref_nsd_nubes_training_development_dataset_xlm_roberta_base_fine_tuned_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|848.2 MB|

## References

https://huggingface.co/ajtamayoh/RE_NegREF_NSD_Nubes_Training_Development_dataset_xlm_RoBERTa_base_fine_tuned

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification