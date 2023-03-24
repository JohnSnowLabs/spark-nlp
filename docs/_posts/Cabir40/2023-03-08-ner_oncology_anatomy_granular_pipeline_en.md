---
layout: model
title: Pipeline to Extract Granular Anatomical Entities from Oncology Texts
author: John Snow Labs
name: ner_oncology_anatomy_granular_pipeline
date: 2023-03-08
tags: [licensed, clinical, en, oncology, ner, anatomy]
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

This pretrained pipeline is built on the top of [ner_oncology_anatomy_granular](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_anatomy_granular_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_granular_pipeline_en_4.3.0_3.2_1678286098380.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_anatomy_granular_pipeline_en_4.3.0_3.2_1678286098380.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_anatomy_granular_pipeline", "en", "clinical/models")

text = '''The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_anatomy_granular_pipeline", "en", "clinical/models")

val text = "The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label   |   confidence |
|---:|:-------------|--------:|------:|:------------|-------------:|
|  0 | left         |      36 |    39 | Direction   |       0.9981 |
|  1 | breast       |      41 |    46 | Site_Breast |       0.9969 |
|  2 | lungs        |      82 |    86 | Site_Lung   |       0.9978 |
|  3 | liver        |      99 |   103 | Site_Liver  |       0.9999 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_anatomy_granular_pipeline|
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