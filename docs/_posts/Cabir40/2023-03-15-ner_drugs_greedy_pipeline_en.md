---
layout: model
title: Pipeline to Detect Drugs - Generalized Single Entity (ner_drugs_greedy)
author: John Snow Labs
name: ner_drugs_greedy_pipeline
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

This pretrained pipeline is built on the top of [ner_drugs_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_greedy_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_drugs_greedy_pipeline_en_4.3.0_3.2_1678877919575.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_drugs_greedy_pipeline_en_4.3.0_3.2_1678877919575.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_drugs_greedy_pipeline", "en", "clinical/models")

text = '''DOSAGE AND ADMINISTRATION The initial dosage of hydrocortisone tablets may vary from 20 mg to 240 mg of hydrocortisone per day depending on the specific disease entity being treated.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_drugs_greedy_pipeline", "en", "clinical/models")

val text = "DOSAGE AND ADMINISTRATION The initial dosage of hydrocortisone tablets may vary from 20 mg to 240 mg of hydrocortisone per day depending on the specific disease entity being treated."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                         |   begin |   end | ner_label   |   confidence |
|---:|:----------------------------------|--------:|------:|:------------|-------------:|
|  0 | hydrocortisone tablets            |      48 |    69 | DRUG        |       0.9923 |
|  1 | 20 mg to 240 mg of hydrocortisone |      85 |   117 | DRUG        |       0.7361 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_drugs_greedy_pipeline|
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