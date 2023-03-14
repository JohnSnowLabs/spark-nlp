---
layout: model
title: Pipeline to Extract Anatomical Entities from Oncology Texts
author: John Snow Labs
name: ner_oncology_anatomy_general_healthcare_pipeline
date: 2023-03-08
tags: [licensed, clinical, oncology, en, ner, anatomy]
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

This pretrained pipeline is built on the top of [ner_oncology_anatomy_general_healthcare](https://nlp.johnsnowlabs.com/2023/01/11/ner_oncology_anatomy_general_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_general_healthcare_pipeline_en_4.3.0_3.2_1678268209780.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_general_healthcare_pipeline_en_4.3.0_3.2_1678268209780.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_anatomy_general_healthcare_pipeline", "en", "clinical/models")

text = "
The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_anatomy_general_healthcare_pipeline", "en", "clinical/models")

val text = "
The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks   |   begin |   end | entities        |   confidence |
|---:|:---------|--------:|------:|:----------------|-------------:|
|  0 | left     |      37 |    40 | Direction       |       0.9948 |
|  1 | breast   |      42 |    47 | Anatomical_Site |       0.5814 |
|  2 | lungs    |      83 |    87 | Anatomical_Site |       0.9486 |
|  3 | liver    |     100 |   104 | Anatomical_Site |       0.9646 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_anatomy_general_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|533.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel