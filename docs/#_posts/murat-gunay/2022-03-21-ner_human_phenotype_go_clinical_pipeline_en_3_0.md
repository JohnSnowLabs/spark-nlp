---
layout: model
title: Pipeline to Detect Normalized Genes and Human Phenotypes
author: John Snow Labs
name: ner_human_phenotype_go_clinical_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, gene, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_human_phenotype_go_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_human_phenotype_go_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_go_clinical_pipeline_en_3.4.1_3.0_1647868376700.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_human_phenotype_go_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("Another disease that shares two of the tumor components of CT, namely GIST and tricarboxylic acid cycle is the Carney-Stratakis syndrome (CSS) or dyad.")
```
```scala
val pipeline = new PretrainedPipeline("ner_human_phenotype_go_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("Another disease that shares two of the tumor components of CT, namely GIST and tricarboxylic acid cycle is the Carney-Stratakis syndrome (CSS) or dyad.")
```
</div>

## Results

```bash
+----+--------------------------+---------+-------+----------+
|    | chunk                    |   begin |   end | entity   |
+====+==========================+=========+=======+==========+
|  0 | tumor                    |      39 |    43 | HP       |
+----+--------------------------+---------+-------+----------+
|  1 | tricarboxylic acid cycle |      79 |   102 | GO       |
+----+--------------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_go_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter