---
layout: model
title: Pipeline to Detect Genomic Variant Information (ner_genetic_variants)
author: John Snow Labs
name: ner_genetic_variants_pipeline
date: 2023-03-14
tags: [ner, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_genetic_variants_pipeline_en_4.3.0_3.2_1678784720653.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_genetic_variants_pipeline_en_4.3.0_3.2_1678784720653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_genetic_variants_pipeline", "en", "clinical/models")

text = '''The mutation pattern of mitochondrial DNA (mtDNA) in mainland Chinese patients with mitochondrial myopathy, encephalopathy, lactic acidosis and stroke-like episodes (MELAS) has been rarely reported, though previous data suggested that the mutation pattern of MELAS could be different among geographically localized populations. We presented the results of comprehensive mtDNA mutation analysis in 92 unrelated Chinese patients with MELAS (85 with classic MELAS and 7 with MELAS/Leigh syndrome (LS) overlap syndrome). The mtDNA A3243G mutation was the most common causal genotype in this patient group (79/92 and 85.9%). The second common gene mutation was G13513A (7/92 and 7.6%). Additionally, we identified T10191C (p.S45P) in ND3, A11470C (p. K237N) in ND4, T13046C (p.M237T) in ND5 and a large-scale deletion (13025-13033:14417-14425) involving partial ND5 and ND6 subunits of complex I in one patient each. Among them, A11470C, T13046C and the single deletion were novel mutations. In summary, patients with mutations affecting mitochondrially encoded complex I (MTND) reached 12.0% (11/92) in this group. It is noteworthy that all seven patients with MELAS/LS overlap syndrome were associated with MTND mutations. Our data emphasize the important role of MTND mutations in the pathogenicity of MELAS, especially MELAS/LS overlap syndrome.PURPOSE: Genes in the complement pathway, including complement factor H (CFH), C2/BF, and C3, have been reported to be associated with age-related macular degeneration (AMD). Genetic variants, single-nucleotide polymorphisms (SNPs), in these genes were geno-typed for a case-control association study in a mainland Han Chinese population. METHODS: One hundred and fifty-eight patients with wet AMD, 80 patients with soft drusen, and 220 matched control subjects were recruited among Han Chinese in mainland China. Seven SNPs in CFH and two SNPs in C2, CFB', and C3 were genotyped using the ABI SNaPshot method. A deletion of 84,682 base pairs covering the CFHR1 and CFHR3 genes was detected by direct polymerase chain reaction and gel electrophoresis. RESULTS: Four SNPs, including rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), in CFH showed a significant association with wet AMD in the cohort of this study. A haplotype containing these four SNPs (CATA) significantly increased protection of wet AMD with a P value of 0.0005 and an odds ratio of 0.29 (95% confidence interval: 0.15-0.60). Unlike in other populations, rs2274700 and rs1410996 did not show a significant association with AMD in the Chinese population of this study. None of the SNPs in CFH showed a significant association with drusen, and none of the SNPs in CFH, C2, CFB, and C3 showed a significant association with either wet AMD or drusen in the cohort of this study. The CFHR1 and CFHR3 deletion was not polymorphic in the Chinese population and was not associated with wet AMD or drusen. CONCLUSION: This study showed that SNPs rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), but not rs7535263, rs1410996, or rs2274700, in CFH were significantly associated with wet AMD in a mainland Han Chinese population. This study showed that CFH was more likely to be AMD susceptibility gene at Chr.1q31 based on the finding that the CFHR1 and CFHR3 deletion was not polymorphic in the cohort of this study, and none of the SNPs that were significantly associated with AMD in a white population in C2, CFB, and C3 genes showed a significant association with AMD.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_genetic_variants_pipeline", "en", "clinical/models")

val text = "The mutation pattern of mitochondrial DNA (mtDNA) in mainland Chinese patients with mitochondrial myopathy, encephalopathy, lactic acidosis and stroke-like episodes (MELAS) has been rarely reported, though previous data suggested that the mutation pattern of MELAS could be different among geographically localized populations. We presented the results of comprehensive mtDNA mutation analysis in 92 unrelated Chinese patients with MELAS (85 with classic MELAS and 7 with MELAS/Leigh syndrome (LS) overlap syndrome). The mtDNA A3243G mutation was the most common causal genotype in this patient group (79/92 and 85.9%). The second common gene mutation was G13513A (7/92 and 7.6%). Additionally, we identified T10191C (p.S45P) in ND3, A11470C (p. K237N) in ND4, T13046C (p.M237T) in ND5 and a large-scale deletion (13025-13033:14417-14425) involving partial ND5 and ND6 subunits of complex I in one patient each. Among them, A11470C, T13046C and the single deletion were novel mutations. In summary, patients with mutations affecting mitochondrially encoded complex I (MTND) reached 12.0% (11/92) in this group. It is noteworthy that all seven patients with MELAS/LS overlap syndrome were associated with MTND mutations. Our data emphasize the important role of MTND mutations in the pathogenicity of MELAS, especially MELAS/LS overlap syndrome.PURPOSE: Genes in the complement pathway, including complement factor H (CFH), C2/BF, and C3, have been reported to be associated with age-related macular degeneration (AMD). Genetic variants, single-nucleotide polymorphisms (SNPs), in these genes were geno-typed for a case-control association study in a mainland Han Chinese population. METHODS: One hundred and fifty-eight patients with wet AMD, 80 patients with soft drusen, and 220 matched control subjects were recruited among Han Chinese in mainland China. Seven SNPs in CFH and two SNPs in C2, CFB', and C3 were genotyped using the ABI SNaPshot method. A deletion of 84,682 base pairs covering the CFHR1 and CFHR3 genes was detected by direct polymerase chain reaction and gel electrophoresis. RESULTS: Four SNPs, including rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), in CFH showed a significant association with wet AMD in the cohort of this study. A haplotype containing these four SNPs (CATA) significantly increased protection of wet AMD with a P value of 0.0005 and an odds ratio of 0.29 (95% confidence interval: 0.15-0.60). Unlike in other populations, rs2274700 and rs1410996 did not show a significant association with AMD in the Chinese population of this study. None of the SNPs in CFH showed a significant association with drusen, and none of the SNPs in CFH, C2, CFB, and C3 showed a significant association with either wet AMD or drusen in the cohort of this study. The CFHR1 and CFHR3 deletion was not polymorphic in the Chinese population and was not associated with wet AMD or drusen. CONCLUSION: This study showed that SNPs rs3753394 (P = 0.0276), rs800292 (P = 0.0266), rs1061170 (P = 0.00514), and rs1329428 (P = 0.0089), but not rs7535263, rs1410996, or rs2274700, in CFH were significantly associated with wet AMD in a mainland Han Chinese population. This study showed that CFH was more likely to be AMD susceptibility gene at Chr.1q31 based on the finding that the CFHR1 and CFHR3 deletion was not polymorphic in the cohort of this study, and none of the SNPs that were significantly associated with AMD in a white population in C2, CFB, and C3 genes showed a significant association with AMD."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label       |   confidence |
|---:|:-------------|--------:|------:|:----------------|-------------:|
|  0 | A3243G       |     527 |   532 | DNAMutation     |       1      |
|  1 | G13513A      |     656 |   662 | DNAMutation     |       0.9994 |
|  2 | T10191C      |     709 |   715 | DNAMutation     |       1      |
|  3 | p.S45P       |     718 |   723 | ProteinMutation |       1      |
|  4 | A11470C      |     734 |   740 | DNAMutation     |       1      |
|  5 | p. K237N     |     743 |   750 | ProteinMutation |       1      |
|  6 | T13046C      |     761 |   767 | DNAMutation     |       1      |
|  7 | p.M237T      |     770 |   776 | ProteinMutation |       1      |
|  8 | A11470C      |     924 |   930 | DNAMutation     |       1      |
|  9 | T13046C      |     933 |   939 | DNAMutation     |       0.9986 |
| 10 | rs3753394    |    2126 |  2134 | SNP             |       1      |
| 11 | rs800292     |    2150 |  2157 | SNP             |       1      |
| 12 | rs1061170    |    2173 |  2181 | SNP             |       1      |
| 13 | rs1329428    |    2202 |  2210 | SNP             |       1      |
| 14 | rs2274700    |    2518 |  2526 | SNP             |       1      |
| 15 | rs1410996    |    2532 |  2540 | SNP             |       1      |
| 16 | rs3753394    |    3000 |  3008 | SNP             |       1      |
| 17 | rs800292     |    3024 |  3031 | SNP             |       1      |
| 18 | rs1061170    |    3047 |  3055 | SNP             |       1      |
| 19 | rs1329428    |    3076 |  3084 | SNP             |       1      |
| 20 | rs7535263    |    3108 |  3116 | SNP             |       1      |
| 21 | rs1410996    |    3119 |  3127 | SNP             |       1      |
| 22 | rs2274700    |    3133 |  3141 | SNP             |       1      |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_genetic_variants_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
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
- NerConverterInternalModel