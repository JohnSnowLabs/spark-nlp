---
layout: model
title: Pipeline to Detect Drug Information
author: John Snow Labs
name: ner_posology_greedy_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, drug, en]
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

This pretrained pipeline is built on the top of [ner_posology_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_greedy_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_greedy_pipeline_en_3.4.1_3.0_1647872438982.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_posology_greedy_pipeline", "en", "clinical/models")

pipeline.annotate("The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.")
```
```scala
val pipeline = new PretrainedPipeline("ner_posology_greedy_pipeline", "en", "clinical/models")

pipeline.annotate("The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.")
```
</div>

## Results

```bash
+----+----------------------------------+---------+-------+------------+
|    | chunks                           |   begin |   end | entities   |
|---:|---------------------------------:|--------:|------:|-----------:|
|  0 | 1 capsule of Advil 10 mg         |      27 |    50 | DRUG       |
|  1 | magnesium hydroxide 100mg/1ml PO |      67 |    98 | DRUG       |
|  2 | for 5 days                       |      52 |    61 | DURATION   |
|  3 | 40 units of insulin glargine     |     168 |   195 | DRUG       |
|  4 | at night                         |     197 |   204 | FREQUENCY  |
|  5 | 12 units of insulin lispro       |     207 |   232 | DRUG       |
|  6 | with meals                       |     234 |   243 | FREQUENCY  |
|  7 | metformin 1000 mg                |     250 |   266 | DRUG       |
|  8 | two times a day                  |     268 |   282 | FREQUENCY  |
+----+----------------------------------+---------+-------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_greedy_pipeline|
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