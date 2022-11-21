---
layout: model
title: Pipeline to Detect Genomic Variants
author: John Snow Labs
name: ner_genetic_variants_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, genomic_variants, en]
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

This pretrained pipeline is built on the top of [ner_genetic_variants](https://nlp.johnsnowlabs.com/2021/06/25/ner_genetic_variants_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_genetic_variants_pipeline_en_3.4.1_3.0_1647872138512.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_genetic_variants_pipeline", "en", "clinical/models")

pipeline.annotate("The mutation pattern of mitochondrial DNA (mtDNA) in mainland Chinese patients with mitochondrial myopathy, encephalopathy, lactic acidosis and stroke-like episodes (MELAS) has been rarely reported, though previous data suggested that the mutation pattern of MELAS could be different among geographically localized populations. We presented the results of comprehensive mtDNA mutation analysis in 92 unrelated Chinese patients with MELAS (85 with classic MELAS and 7 with MELAS/Leigh syndrome (LS) overlap syndrome). The mtDNA A3243G mutation was the most common causal genotype in this patient group (79/92 and 85.9%). The second common gene mutation was G13513A (7/92 and 7.6%). Additionally, we identified T10191C (p.S45P) in ND3, A11470C (p. K237N) in ND4, T13046C (p.M237T) in ND5 and a large-scale deletion (13025-13033:14417-14425) involving partial ND5 and ND6 subunits of complex I in one patient each. Among them, A11470C, T13046C and the single deletion were novel mutations. In summary, patients with mutations affecting mitochondrially encoded complex I (MTND) reached 12.0% (11/92) in this group. It is noteworthy that all seven patients with MELAS/LS overlap syndrome were associated with MTND mutations. Our data emphasize the important role of MTND mutations in the pathogenicity of MELAS, especially MELAS/LS overlap syndrome.PURPOSE: Genes in the complement pathway, including complement factor H (CFH), C2/BF, and C3, have been reported to be associated with age-related macular degeneration (AMD). Genetic variants, single-nucleotide polymorphisms (SNPs), in these genes were geno-typed for a case-control association study in a mainland Han Chinese population. METHODS: One hundred and fifty-eight patients with wet AMD, 80 patients with soft drusen, and 220 matched control subjects were recruited among Han Chinese in mainland China. Seven SNPs in CFH and two SNPs in C2, CFB', and C3 were genotyped using the ABI SNaPshot method. A deletion of 84,682 base pairs covering the CFHR1 and CFHR3 genes was detected by direct polymerase chain reaction and gel electrophoresis. RESULTS: Four SNPs, including rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), in CFH showed a significant association with wet AMD in the cohort of this study. A haplotype containing these four SNPs (CATA) significantly increased protection of wet AMD with a P value of 0.0005 and an odds ratio of 0.29 (95% confidence interval: 0.15-0.60). Unlike in other populations, rs2274700 and rs1410996 did not show a significant association with AMD in the Chinese population of this study. None of the SNPs in CFH showed a significant association with drusen, and none of the SNPs in CFH, C2, CFB, and C3 showed a significant association with either wet AMD or drusen in the cohort of this study. The CFHR1 and CFHR3 deletion was not polymorphic in the Chinese population and was not associated with wet AMD or drusen. CONCLUSION: This study showed that SNPs rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), but not rs7535263, rs1410996, or rs2274700, in CFH were significantly associated with wet AMD in a mainland Han Chinese population. This study showed that CFH was more likely to be AMD susceptibility gene at Chr.1q31 based on the finding that the CFHR1 and CFHR3 deletion was not polymorphic in the cohort of this study, and none of the SNPs that were significantly associated with AMD in a white population in C2, CFB, and C3 genes showed a significant association with AMD.")
```
```scala
val pipeline = new PretrainedPipeline("ner_genetic_variants_pipeline", "en", "clinical/models")

pipeline.annotate("The mutation pattern of mitochondrial DNA (mtDNA) in mainland Chinese patients with mitochondrial myopathy, encephalopathy, lactic acidosis and stroke-like episodes (MELAS) has been rarely reported, though previous data suggested that the mutation pattern of MELAS could be different among geographically localized populations. We presented the results of comprehensive mtDNA mutation analysis in 92 unrelated Chinese patients with MELAS (85 with classic MELAS and 7 with MELAS/Leigh syndrome (LS) overlap syndrome). The mtDNA A3243G mutation was the most common causal genotype in this patient group (79/92 and 85.9%). The second common gene mutation was G13513A (7/92 and 7.6%). Additionally, we identified T10191C (p.S45P) in ND3, A11470C (p. K237N) in ND4, T13046C (p.M237T) in ND5 and a large-scale deletion (13025-13033:14417-14425) involving partial ND5 and ND6 subunits of complex I in one patient each. Among them, A11470C, T13046C and the single deletion were novel mutations. In summary, patients with mutations affecting mitochondrially encoded complex I (MTND) reached 12.0% (11/92) in this group. It is noteworthy that all seven patients with MELAS/LS overlap syndrome were associated with MTND mutations. Our data emphasize the important role of MTND mutations in the pathogenicity of MELAS, especially MELAS/LS overlap syndrome.PURPOSE: Genes in the complement pathway, including complement factor H (CFH), C2/BF, and C3, have been reported to be associated with age-related macular degeneration (AMD). Genetic variants, single-nucleotide polymorphisms (SNPs), in these genes were geno-typed for a case-control association study in a mainland Han Chinese population. METHODS: One hundred and fifty-eight patients with wet AMD, 80 patients with soft drusen, and 220 matched control subjects were recruited among Han Chinese in mainland China. Seven SNPs in CFH and two SNPs in C2, CFB', and C3 were genotyped using the ABI SNaPshot method. A deletion of 84,682 base pairs covering the CFHR1 and CFHR3 genes was detected by direct polymerase chain reaction and gel electrophoresis. RESULTS: Four SNPs, including rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), in CFH showed a significant association with wet AMD in the cohort of this study. A haplotype containing these four SNPs (CATA) significantly increased protection of wet AMD with a P value of 0.0005 and an odds ratio of 0.29 (95% confidence interval: 0.15-0.60). Unlike in other populations, rs2274700 and rs1410996 did not show a significant association with AMD in the Chinese population of this study. None of the SNPs in CFH showed a significant association with drusen, and none of the SNPs in CFH, C2, CFB, and C3 showed a significant association with either wet AMD or drusen in the cohort of this study. The CFHR1 and CFHR3 deletion was not polymorphic in the Chinese population and was not associated with wet AMD or drusen. CONCLUSION: This study showed that SNPs rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), but not rs7535263, rs1410996, or rs2274700, in CFH were significantly associated with wet AMD in a mainland Han Chinese population. This study showed that CFH was more likely to be AMD susceptibility gene at Chr.1q31 based on the finding that the CFHR1 and CFHR3 deletion was not polymorphic in the cohort of this study, and none of the SNPs that were significantly associated with AMD in a white population in C2, CFB, and C3 genes showed a significant association with AMD.")
```
</div>

## Results

```bash
+---------+---------------+
|chunk    |ner_label      |
+---------+---------------+
|A3243G   |DNAMutation    |
|G13513A  |DNAMutation    |
|T10191C  |DNAMutation    |
|p.S45P   |ProteinMutation|
|A11470C  |DNAMutation    |
|p. K237N |ProteinMutation|
|T13046C  |DNAMutation    |
|p.       |ProteinMutation|
|M237T    |ProteinMutation|
|A11470C  |DNAMutation    |
|T13046C  |DNAMutation    |
|rs3753394|SNP            |
|rs800292 |SNP            |
|rs1061170|SNP            |
|rs1329428|SNP            |
|rs2274700|SNP            |
|rs1410996|SNP            |
|rs3753394|SNP            |
|rs800292 |SNP            |
|rs1061170|SNP            |
|rs1329428|SNP            |
|rs7535263|SNP            |
|rs1410996|SNP            |
|rs2274700|SNP            |
+---------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_genetic_variants_pipeline|
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