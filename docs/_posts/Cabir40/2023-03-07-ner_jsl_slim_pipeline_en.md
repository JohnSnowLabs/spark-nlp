---
layout: model
title: Pipeline to Detect Clinical Entities (ner_jsl_slim)
author: John Snow Labs
name: ner_jsl_slim_pipeline
date: 2023-03-07
tags: [ner, en, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.1
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_jsl_slim](https://nlp.johnsnowlabs.com/2021/08/13/ner_jsl_slim_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_slim_pipeline_en_4.3.1_3.2_1678195679312.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_slim_pipeline_en_4.3.1_3.2_1678195679312.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_jsl_slim_pipeline", "en", "clinical/models")

text = "Hyperparathyroidism was considered upon the fourth occasion. The history of weakness and generalized joint pains were present. He also had history of epigastric pain diagnosed informally as gastritis. He had previously had open reduction and internal fixation for the initial two fractures under general anesthesia. He sustained mandibular fracture."

result = pipeline.fullAnnotate(text)

```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_jsl_slim_pipeline", "en", "clinical/models")

val text = "Hyperparathyroidism was considered upon the fourth occasion. The history of weakness and generalized joint pains were present. He also had history of epigastric pain diagnosed informally as gastritis. He had previously had open reduction and internal fixation for the initial two fractures under general anesthesia. He sustained mandibular fracture."

val result = pipeline.fullAnnotate(text)


```
</div>

## Results

```bash
|    | chunks                               |   begin |   end | entities                  |   confidence |
|---:|:-------------------------------------|--------:|------:|:--------------------------|-------------:|
|  0 | Hyperparathyroidism                  |       0 |    18 | Disease_Syndrome_Disorder |     0.9977   |
|  1 | weakness                             |      76 |    83 | Symptom                   |     0.9744   |
|  2 | generalized joint pains              |      89 |   111 | Symptom                   |     0.584067 |
|  3 | He                                   |     127 |   128 | Demographics              |     0.9996   |
|  4 | epigastric pain                      |     150 |   164 | Symptom                   |     0.66655  |
|  5 | gastritis                            |     190 |   198 | Disease_Syndrome_Disorder |     0.9874   |
|  6 | He                                   |     201 |   202 | Demographics              |     0.9995   |
|  7 | open reduction and internal fixation |     223 |   258 | Procedure                 |     0.61648  |
|  8 | fractures under general anesthesia   |     280 |   313 | Drug                      |     0.79585  |
|  9 | He                                   |     316 |   317 | Demographics              |     0.9992   |
| 10 | sustained mandibular fracture        |     319 |   347 | Disease_Syndrome_Disorder |     0.662467 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_slim_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.1+|
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