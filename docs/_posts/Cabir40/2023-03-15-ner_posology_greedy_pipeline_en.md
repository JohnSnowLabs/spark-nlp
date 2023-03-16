---
layout: model
title: Pipeline to Detect Drugs and Posology Entities (ner_posology_greedy)
author: John Snow Labs
name: ner_posology_greedy_pipeline
date: 2023-03-15
tags: [ner, clinical, licensed, en]
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

This pretrained pipeline is built on the top of [ner_posology_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_greedy_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_greedy_pipeline_en_4.3.0_3.2_1678869761403.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_greedy_pipeline_en_4.3.0_3.2_1678869761403.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_posology_greedy_pipeline", "en", "clinical/models")

text = '''The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_posology_greedy_pipeline", "en", "clinical/models")

val text = "The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                   |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------------------------|--------:|------:|:------------|-------------:|
|  0 | 1 capsule of Advil 10 mg                    |      27 |    50 | DRUG        |     0.638183 |
|  1 | for 5 days                                  |      52 |    61 | DURATION    |     0.573533 |
|  2 | magnesium hydroxide 100mg/1ml suspension PO |      67 |   109 | DRUG        |     0.68788  |
|  3 | 40 units of insulin glargine                |     179 |   206 | DRUG        |     0.61964  |
|  4 | at night                                    |     208 |   215 | FREQUENCY   |     0.7431   |
|  5 | 12 units of insulin lispro                  |     218 |   243 | DRUG        |     0.66034  |
|  6 | with meals                                  |     245 |   254 | FREQUENCY   |     0.79235  |
|  7 | metformin 1000 mg                           |     261 |   277 | DRUG        |     0.707133 |
|  8 | two times a day                             |     279 |   293 | FREQUENCY   |     0.700825 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_greedy_pipeline|
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