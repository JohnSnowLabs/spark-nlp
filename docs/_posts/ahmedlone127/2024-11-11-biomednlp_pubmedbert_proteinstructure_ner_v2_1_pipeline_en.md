---
layout: model
title: English biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline pipeline BertForTokenClassification from PDBEurope
author: John Snow Labs
name: biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline
date: 2024-11-11
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

Pretrained BertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline` is a English model originally trained by PDBEurope.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline_en_5.5.1_3.0_1731285765036.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline_en_5.5.1_3.0_1731285765036.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|biomednlp_pubmedbert_proteinstructure_ner_v2_1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.3 MB|

## References

https://huggingface.co/PDBEurope/BiomedNLP-PubMedBERT-ProteinStructure-NER-v2.1

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForTokenClassification