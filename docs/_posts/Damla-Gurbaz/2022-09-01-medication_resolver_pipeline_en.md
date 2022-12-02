---
layout: model
title: Pipeline to Resolve Medication Codes
author: John Snow Labs
name: medication_resolver_pipeline
date: 2022-09-01
tags: [resolver, snomed, umls, rxnorm, ndc, ade, en, licensed, pipeline]
task: Pipeline Healthcare
language: en
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pretrained resolver pipeline to extract medications and resolve their adverse reactions (ADE), RxNorm, UMLS, NDC, SNOMED CT codes, and action/treatments in clinical text.

Action/treatments are available for branded medication, and SNOMED codes are available for non-branded medication.

This pipeline can be used as Lightpipeline (with `annotate/fullAnnotate`). You can use `medication_resolver_transform_pipeline` for Spark transform.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/medication_resolver_pipeline_en_4.0.2_3.0_1662044306623.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

med_resolver_pipeline = PretrainedPipeline("medication_resolver_pipeline", "en", "clinical/models")

text = """The patient was prescribed Amlodopine Vallarta 10-320mg, Eviplera. The other patient is given Lescol 40 MG and Everolimus 1.5 mg tablet."""

result = med_resolver_pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val med_resolver_pipeline = new PretrainedPipeline("medication_resolver_pipeline", "en", "clinical/models")

val result = med_resolver_pipeline.fullAnnotate("""The patient was prescribed Amlodopine Vallarta 10-320mg, Eviplera. The other patient is given Lescol 40 MG and Everolimus 1.5 mg tablet.""")
```
</div>

## Results

```bash
|    | chunks                       | entities   | ADE                         |   RxNorm | Action                     | Treatment                                  | UMLS     | SNOMED_CT   | NDC_Product   | NDC_Package   |
|---:|:-----------------------------|:-----------|:----------------------------|---------:|:---------------------------|:-------------------------------------------|:---------|:------------|:--------------|:--------------|
|  0 | Amlodopine Vallarta 10-320mg | DRUG       | Gynaecomastia               |   722131 | NONE                       | NONE                                       | C1949334 | 425838008   | 00093-7693    | 00093-7693-56 |
|  1 | Eviplera                     | DRUG       | Anxiety                     |   217010 | Inhibitory Bone Resorption | Osteoporosis                               | C0720318 | NONE        | NONE          | NONE          |
|  2 | Lescol 40 MG                 | DRUG       | NONE                        |   103919 | Hypocholesterolemic        | Heterozygous Familial Hypercholesterolemia | C0353573 | NONE        | 00078-0234    | 00078-0234-05 |
|  3 | Everolimus 1.5 mg tablet     | DRUG       | Acute myocardial infarction |  2056895 | NONE                       | NONE                                       | C4723581 | NONE        | 00054-0604    | 00054-0604-21 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|medication_resolver_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.0.2+|
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
- ChunkMapperModel
- ChunkMapperFilterer
- Chunk2Doc
- BertSentenceEmbeddings
- SentenceEntityResolverModel
- ResolverMerger
- ResolverMerger
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperModel
- ChunkMapperModel
- Finisher
