---
layout: model
title: Pipeline to Detect Problem, Test and Treatment (Large)
author: John Snow Labs
name: ner_clinical_large_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, problem, test, treatment, en]
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

This pretrained pipeline is built on the top of [ner_clinical_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_clinical_large_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_large_pipeline_en_3.4.1_3.0_1647872951545.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_large_pipeline_en_3.4.1_3.0_1647872951545.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_clinical_large_pipeline", "en", "clinical/models")

pipeline.annotate("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes.")
```
```scala
val pipeline = new PretrainedPipeline("ner_clinical_large_pipeline", "en", "clinical/models")

pipeline.annotate("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes.")
```
</div>

## Results

```bash
+-----------------------------------------------------------+---------+
|the G-protein-activated inwardly rectifying potassium (GIRK|TREATMENT|
|the genomicorganization                                    |TREATMENT|
|a candidate gene forType II diabetes mellitus              |PROBLEM  |
|byapproximately                                            |TREATMENT|
|single nucleotide polymorphisms                            |TREATMENT|
|aVal366Ala substitution                                    |TREATMENT|
|an 8 base-pair                                             |TREATMENT|
|insertion/deletion                                         |PROBLEM  |
|Ourexpression studies                                      |TEST     |
|the transcript in various humantissues                     |PROBLEM  |
|fat andskeletal muscle                                     |PROBLEM  |
|furtherstudies                                             |PROBLEM  |
|the KCNJ9 protein                                          |TREATMENT|
|evaluation                                                 |TEST     |
|Type II diabetes                                           |PROBLEM  |
|the treatment                                              |TREATMENT|
|breast cancer                                              |PROBLEM  |
|the standard therapy                                       |TREATMENT|
|anthracyclines                                             |TREATMENT|
|taxanes                                                    |TREATMENT|
+-----------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_large_pipeline|
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