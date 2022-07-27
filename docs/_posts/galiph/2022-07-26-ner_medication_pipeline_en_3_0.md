---
layout: model
title: Pipeline for Detect Medication
author: John Snow Labs
name: ner_medication_pipeline
date: 2022-07-26
tags: [ner, en, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

A pretrained pipeline to detect medication entities. It was built on the top of `ner_posology_greedy` model and also augmented with the drug names mentioned in UK and US drugbank datasets.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_medication_pipeline_en_4.0.0_3.0_1658843236915.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

ner_medication_pipeline = PretrainedPipeline("ner_medication_pipeline", "en", "clinical/models")

text = """The patient was prescribed metformin 1000 MG, and glipizide 2.5 MG. The other patient was given Fragmin 5000 units, Xenaderm to wounds topically b.i.d. and OxyContin 30 mg."""

result = ner_medication_pipeline.fullAnnotate([text])
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val ner_medication_pipeline = new PretrainedPipeline("ner_medication_pipeline", "en", "clinical/models")

val result = ner_medication_pipeline.fullAnnotate("The patient was prescribed metformin 1000 MG, and glipizide 2.5 MG. The other patient was given Fragmin 5000 units, Xenaderm to wounds topically b.i.d. and OxyContin 30 mg."")(0)
```
</div>

## Results

```bash
| ner_chunk          | entity   |
|:-------------------|:---------|
| metformin 1000 MG  | DRUG     |
| glipizide 2.5 MG   | DRUG     |
| Fragmin 5000 units | DRUG     |
| Xenaderm           | DRUG     |
| OxyContin 30 mg    | DRUG     |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_medication_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 4.0.0+|
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
- TextMatcherModel
- ChunkMergeModel
- Finisher
