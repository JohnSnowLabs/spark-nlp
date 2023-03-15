---
layout: model
title: Pipeline to Extract conditions and benefits from drug reviews
author: John Snow Labs
name: ner_supplement_clinical_pipeline
date: 2023-03-14
tags: [licensed, ner, en, clinical]
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

This pretrained pipeline is built on the top of [ner_supplement_clinical](https://nlp.johnsnowlabs.com/2022/02/01/ner_supplement_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_supplement_clinical_pipeline_en_4.3.0_3.2_1678777179236.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_supplement_clinical_pipeline_en_4.3.0_3.2_1678777179236.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_supplement_clinical_pipeline", "en", "clinical/models")

text = '''Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_supplement_clinical_pipeline", "en", "clinical/models")

val text = "Excellent!. The state of health improves, nervousness disappears, and night sleep improves. It also promotes hair and nail growth. I recommend :"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks   |   begin |   end | ner_label   |   confidence |
|---:|:-------------|--------:|------:|:------------|-------------:|
|  0 | nervousness  |      42 |    52 | CONDITION   |      0.9999  |
|  1 | night sleep  |      70 |    80 | BENEFIT     |      0.80775 |
|  2 | hair         |     109 |   112 | BENEFIT     |      0.9997  |
|  3 | nail growth  |     118 |   128 | BENEFIT     |      0.9997  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_supplement_clinical_pipeline|
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