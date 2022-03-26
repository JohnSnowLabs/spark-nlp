---
layout: model
title: Pipeline to Detect biological concepts (biobert)
author: John Snow Labs
name: ner_bionlp_biobert_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_bionlp_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_bionlp_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_biobert_pipeline_en_3.4.1_3.0_1647871746678.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_bionlp_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.")
```
```scala
val pipeline = new PretrainedPipeline("ner_bionlp_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.")
```
</div>

## Results

```bash
|    | chunk             | entity        |
|---:|:------------------|:--------------|
|  0 | median            | Duration      |
|  1 | overall survival  | End_Point     |
|  2 | with              | Trial_Group   |
|  3 | without topotecan | Trial_Group   |
|  4 | 4.0               | Value         |
|  5 | 3.6 months        | Value         |
|  6 | 23                | Patient_Count |
|  7 | 63                | Patient_Count |
|  8 | 55                | Patient_Count |
|  9 | 33 patients       | Patient_Count |
| 10 | topotecan         | Trial_Group   |
| 11 | 11                | Patient_Count |
| 12 | 61                | Patient_Count |
| 13 | 66                | Patient_Count |
| 14 | 32 patients       | Patient_Count |
| 15 | without topotecan | Trial_Group   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bionlp_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
