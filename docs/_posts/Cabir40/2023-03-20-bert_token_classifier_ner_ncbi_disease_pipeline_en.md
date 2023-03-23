---
layout: model
title: Pipeline to Detect Diseases in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_ncbi_disease_pipeline
date: 2023-03-20
tags: [en, ner, clinical, licensed, bertfortokenclassification]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_ncbi_disease](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_ncbi_disease_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_ncbi_disease_pipeline_en_4.3.0_3.2_1679303325122.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_ncbi_disease_pipeline_en_4.3.0_3.2_1679303325122.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_ncbi_disease_pipeline", "en", "clinical/models")

text = '''Kniest dysplasia is a moderately severe type II collagenopathy, characterized by short trunk and limbs, kyphoscoliosis, midface hypoplasia, severe myopia, and hearing loss.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_ncbi_disease_pipeline", "en", "clinical/models")

val text = "Kniest dysplasia is a moderately severe type II collagenopathy, characterized by short trunk and limbs, kyphoscoliosis, midface hypoplasia, severe myopia, and hearing loss."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk              |   begin |   end | ner_label   |   confidence |
|---:|:-----------------------|--------:|------:|:------------|-------------:|
|  0 | Kniest dysplasia       |       0 |    15 | Disease     |     0.999886 |
|  1 | type II collagenopathy |      40 |    61 | Disease     |     0.999934 |
|  2 | kyphoscoliosis         |     104 |   117 | Disease     |     0.99994  |
|  3 | midface hypoplasia     |     120 |   137 | Disease     |     0.999911 |
|  4 | myopia                 |     147 |   152 | Disease     |     0.999894 |
|  5 | hearing loss           |     159 |   170 | Disease     |     0.999351 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_ncbi_disease_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel