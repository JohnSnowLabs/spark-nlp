---
layout: model
title: Pipeline to Detect Entities Related to Cancer Diagnosis
author: John Snow Labs
name: ner_oncology_diagnosis_pipeline
date: 2023-03-09
tags: [licensed, clinical, en, ner, oncology]
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

This pretrained pipeline is built on the top of [ner_oncology_diagnosis](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_diagnosis_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_diagnosis_pipeline_en_4.3.0_3.2_1678346464717.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_diagnosis_pipeline_en_4.3.0_3.2_1678346464717.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_diagnosis_pipeline", "en", "clinical/models")

text = '''Two years ago, the patient presented with a tumor in her left breast and adenopathies. She was diagnosed with invasive ductal carcinoma. Last week she was also found to have a lung metastasis.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_diagnosis_pipeline", "en", "clinical/models")

val text = "Two years ago, the patient presented with a tumor in her left breast and adenopathies. She was diagnosed with invasive ductal carcinoma. Last week she was also found to have a lung metastasis."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks   |   begin |   end | ner_label         |   confidence |
|---:|:-------------|--------:|------:|:------------------|-------------:|
|  0 | tumor        |      44 |    48 | Tumor_Finding     |       0.9958 |
|  1 | adenopathies |      73 |    84 | Adenopathy        |       0.6287 |
|  2 | invasive     |     110 |   117 | Histological_Type |       0.9965 |
|  3 | ductal       |     119 |   124 | Histological_Type |       0.9996 |
|  4 | carcinoma    |     126 |   134 | Cancer_Dx         |       0.9988 |
|  5 | metastasis   |     181 |   190 | Metastasis        |       0.9996 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_diagnosis_pipeline|
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