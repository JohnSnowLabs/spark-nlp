---
layout: model
title: Pipeline to Extract Mentions of Response to Cancer Treatment
author: John Snow Labs
name: ner_oncology_response_to_treatment_pipeline
date: 2023-03-09
tags: [licensed, clinical, en, oncology, ner, treatment]
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

This pretrained pipeline is built on the top of [ner_oncology_response_to_treatment](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_response_to_treatment_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_response_to_treatment_pipeline_en_4.3.0_3.2_1678349824229.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_response_to_treatment_pipeline_en_4.3.0_3.2_1678349824229.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_response_to_treatment_pipeline", "en", "clinical/models")

text = '''She completed her first-line therapy, but some months later there was recurrence of the breast cancer.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_response_to_treatment_pipeline", "en", "clinical/models")

val text = "She completed her first-line therapy, but some months later there was recurrence of the breast cancer."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label             |   confidence |
|---:|:-------------|--------:|------:|:----------------------|-------------:|
|  0 | recurrence   |      70 |    79 | Response_To_Treatment |       0.9767 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_response_to_treatment_pipeline|
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