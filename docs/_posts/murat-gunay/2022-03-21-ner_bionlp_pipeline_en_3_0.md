---
layout: model
title: Pipeline to Detect Cancer Genetics
author: John Snow Labs
name: ner_bionlp_pipeline
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

This pretrained pipeline is built on the top of [ner_bionlp](https://nlp.johnsnowlabs.com/2021/03/31/ner_bionlp_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_pipeline_en_3.4.1_3.0_1647871349979.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_bionlp_pipeline", "en", "clinical/models")


pipeline.annotate("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.")
```
```scala
val pipeline = new PretrainedPipeline("ner_bionlp_pipeline", "en", "clinical/models")


pipeline.annotate("The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.")
```
</div>

## Results

```bash
|id |sentence_id|chunk                 |begin|end|ner_label           |
+---+-----------+----------------------+-----+---+--------------------+
|0  |0          |human                 |4    |8  |Organism            |
|0  |0          |Kir 3.3               |17   |23 |Gene_or_gene_product|
|0  |0          |GIRK3                 |26   |30 |Gene_or_gene_product|
|0  |0          |potassium             |92   |100|Simple_chemical     |
|0  |0          |GIRK                  |103  |106|Gene_or_gene_product|
|0  |1          |chromosome 1q21-23    |188  |205|Cellular_component  |
|0  |5          |pancreas              |697  |704|Organ               |
|0  |5          |tissues               |740  |746|Tissue              |
|0  |5          |fat andskeletal muscle|749  |770|Tissue              |
|0  |6          |KCNJ9                 |801  |805|Gene_or_gene_product|
|0  |6          |Type II               |940  |946|Gene_or_gene_product|
|1  |0          |breast cancer         |84   |96 |Cancer              |
|1  |0          |patients              |134  |141|Organism            |
|1  |0          |anthracyclines        |167  |180|Simple_chemical     |
|1  |0          |taxanes               |186  |192|Simple_chemical     |
|1  |1          |vinorelbine           |246  |256|Simple_chemical     |
|1  |1          |patients              |273  |280|Organism            |
|1  |1          |breast                |309  |314|Cancer              |
|1  |1          |vinorelbine inpatients|386  |407|Simple_chemical     |
|1  |1          |anthracyclines        |433  |446|Simple_chemical     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bionlp_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
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