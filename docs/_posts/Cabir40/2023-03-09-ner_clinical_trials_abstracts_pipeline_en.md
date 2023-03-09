---
layout: model
title: Pipeline to Extract entities in clinical trial abstracts
author: John Snow Labs
name: ner_clinical_trials_abstracts_pipeline
date: 2023-03-09
tags: [ner, clinical, en, licensed]
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

This pretrained pipeline is built on the top of [ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/06/22/ner_clinical_trials_abstracts_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_pipeline_en_4.3.0_3.2_1678386393248.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_pipeline_en_4.3.0_3.2_1678386393248.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

text = '''A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "en", "clinical/models")

val text = "A one-year, randomised, multicentre trial comparing insulin glargine with NPH insulin in combination with oral agents in patients with type 2 diabetes. In a multicentre, open, randomised study, 570 patients with Type 2 diabetes, aged 34 - 80 years, were treated for 52 weeks with insulin glargine or NPH insulin given once daily at bedtime."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks       |   begin |   end | ner_label          |   confidence |
|---:|:-----------------|--------:|------:|:-------------------|-------------:|
|  0 | randomised       |      12 |    21 | CTDesign           |     0.9996   |
|  1 | multicentre      |      24 |    34 | CTDesign           |     0.9998   |
|  2 | insulin glargine |      52 |    67 | Drug               |     0.99135  |
|  3 | NPH insulin      |      74 |    84 | Drug               |     0.9687   |
|  4 | type 2 diabetes  |     135 |   149 | DisorderOrSyndrome |     0.999933 |
|  5 | multicentre      |     157 |   167 | CTDesign           |     0.9997   |
|  6 | open             |     170 |   173 | CTDesign           |     0.9988   |
|  7 | randomised       |     176 |   185 | CTDesign           |     0.9984   |
|  8 | 570              |     194 |   196 | NumberPatients     |     0.9906   |
|  9 | Type 2 diabetes  |     212 |   226 | DisorderOrSyndrome |     0.9999   |
| 10 | 34               |     234 |   235 | Age                |     0.9999   |
| 11 | 80               |     239 |   240 | Age                |     0.9931   |
| 12 | 52 weeks         |     266 |   273 | Duration           |     0.9794   |
| 13 | insulin glargine |     280 |   295 | Drug               |     0.989    |
| 14 | NPH insulin      |     300 |   310 | Drug               |     0.97955  |
| 15 | once daily       |     318 |   327 | DrugTime           |     0.999    |
| 16 | bedtime          |     332 |   338 | DrugTime           |     0.9937   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_trials_abstracts_pipeline|
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