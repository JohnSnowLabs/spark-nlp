---
layout: model
title: Pipeline to Resolve Medication Codes
author: John Snow Labs
name: medication_resolver_pipeline
date: 2022-08-01
tags: [resolver, rxnorm, umls, snomed, ndc, en, licensed]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pretrained resolver pipeline to extract medications and resolve RxNorm, UMLS, NDC, SNOMED CT codes, and action/treatments in clinical text.

Action/treatments are available for branded medication, and SNOMED codes are available for non-branded medication.

This pipeline can be used as Lightpipeline (with `annotate/fullAnnotate`). You can use `medication_resolver_transform_pipeline` for Spark transform.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/medication_resolver_pipeline_en_4.0.0_3.0_1659376739181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

med_resolver_pipeline = PretrainedPipeline("medication_resolver_pipeline", "en", "clinical/models")

text = """The patient was prescribed Mycobutn 150 MG, Salagen 5 MG oral tablet, 
The other patient is given Lescol 40 MG and Lidoderm 0.05 MG/MG, triazolam 0.125 MG Oral Tablet, metformin hydrochloride 1000 MG Oral Tablet"""

result = med_resolver_pipeline.fullAnnotate([text])
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val med_resolver_pipeline = new PretrainedPipeline("medication_resolver_pipeline", "en", "clinical/models")

val result = med_resolver_pipeline.fullAnnotate("The patient was prescribed Mycobutn 150 MG, Salagen 5 MG oral tablet, 
The other patient is given Lescol 40 MG and Lidoderm 0.05 MG/MG, triazolam 0.125 MG Oral Tablet, metformin hydrochloride 1000 MG Oral Tablet"")
```

{:.nlu-block}
```python
|    | ner_chunk                                   |   RxNorm_Chunk | Action              | Treatment                                  | UMLS     | SNOMED_CT   | NDC_Product   | NDC_Package   | entity   |
|---:|:--------------------------------------------|---------------:|:--------------------|:-------------------------------------------|:---------|:------------|:--------------|:--------------|:---------|
|  0 | Mycobutn 150 MG                             |         103899 | Antimiycobacterials | Infection                                  | C0353536 | -           | 00013-5301    | 00013-5301-17 | DRUG     |
|  1 | Salagen 5 MG oral tablet                    |        1000915 | Antiglaucomatous    | Cancer                                     | C0361693 | -           | 59212-0705    | 59212-0705-10 | DRUG     |
|  2 | Lescol 40 MG                                |         103919 | Hypocholesterolemic | Heterozygous Familial Hypercholesterolemia | C0353573 | -           | 00078-0234    | 00078-0234-05 | DRUG     |
|  3 | Lidoderm 0.05 MG/MG                         |        1011705 | Anesthetic          | Pain                                       | C0875706 | -           | 00247-2129    | 00247-2129-30 | DRUG     |
|  4 | triazolam 0.125 MG Oral Tablet              |         198317 | -                   | -                                          | C0690642 | 373981005   | 00054-4858    | 00054-4858-25 | DRUG     |
|  5 | metformin hydrochloride 1000 MG Oral Tablet |         861004 | -                   | -                                          | C0978482 | 376701008   | 00093-7214    | 00185-0221-01 | DRUG     |

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|medication_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|3.1 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel
- TextMatcherModel
- ChunkMergeModel
- ChunkMapperModel
- ChunkMapperFilterer
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- ResolverMerger
- ResolverMerger
- ChunkMapperModel
- Finisher
