---
layout: model
title: Pipeline to Detect Chemical Compounds and Genes
author: John Snow Labs
name: ner_chemprot_clinical_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
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

This pretrained pipeline is built on the top of [ner_chemprot_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_chemprot_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemprot_clinical_pipeline_en_3.4.1_3.0_1647868192988.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_chemprot_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.")
```
```scala
val pipeline = new PretrainedPipeline("ner_chemprot_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("Keratinocyte growth factor and acidic fibroblast growth factor are mitogens for primary cultures of mammary epithelium.")
```
</div>

## Results

```bash
+----+---------------------------------+---------+-------+----------+
|    | chunk                           |   begin |   end | entity   |
+====+=================================+=========+=======+==========+
|  0 | Keratinocyte growth factor      |       0 |    25 | GENE-Y   |
+----+---------------------------------+---------+-------+----------+
|  1 | acidic fibroblast growth factor |      31 |    61 | GENE-Y   |
+----+---------------------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemprot_clinical_pipeline|
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