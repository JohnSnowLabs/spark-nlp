---
layout: model
title: Pipeline to Detect Genes and Human Phenotypes
author: John Snow Labs
name: ner_human_phenotype_gene_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_human_phenotype_gene_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_gene_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_biobert_pipeline_en_3.4.1_3.0_1647867336282.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_human_phenotype_gene_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).")
```
```scala
val pipeline = new PretrainedPipeline("ner_human_phenotype_gene_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).")
```
</div>

## Results

```bash
+----------------+--------+
|chunks          |entities|
+----------------+--------+
|type            |GENE    |
|polyhydramnios  |HP      |
|polyuria        |HP      |
|nephrocalcinosis|HP      |
|hypokalemia     |HP      |
+----------------+--------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_gene_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
