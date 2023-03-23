---
layout: model
title: Pipeline to Detect Chemicals in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc4chemd_chemicals_pipeline
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bc4chemd_chemicals](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc4chemd_chemicals_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc4chemd_chemicals_pipeline_en_4.3.0_3.2_1679301435930.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc4chemd_chemicals_pipeline_en_4.3.0_3.2_1679301435930.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_bc4chemd_chemicals_pipeline", "en", "clinical/models")

text = '''The main isolated compounds were triterpenes (alpha - amyrin, beta - amyrin, lupeol, betulin, betulinic acid, uvaol, erythrodiol and oleanolic acid) and phenolic acid derivatives from 4 - hydroxybenzoic acid (gallic and protocatechuic acids and isocorilagin).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bc4chemd_chemicals_pipeline", "en", "clinical/models")

val text = "The main isolated compounds were triterpenes (alpha - amyrin, beta - amyrin, lupeol, betulin, betulinic acid, uvaol, erythrodiol and oleanolic acid) and phenolic acid derivatives from 4 - hydroxybenzoic acid (gallic and protocatechuic acids and isocorilagin)."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                       |   begin |   end | ner_label   |   confidence |
|---:|:--------------------------------|--------:|------:|:------------|-------------:|
|  0 | triterpenes                     |      33 |    43 | CHEM        |     0.99999  |
|  1 | alpha - amyrin                  |      46 |    59 | CHEM        |     0.999939 |
|  2 | beta - amyrin                   |      62 |    74 | CHEM        |     0.999679 |
|  3 | lupeol                          |      77 |    82 | CHEM        |     0.999968 |
|  4 | betulin                         |      85 |    91 | CHEM        |     0.999975 |
|  5 | betulinic acid                  |      94 |   107 | CHEM        |     0.999984 |
|  6 | uvaol                           |     110 |   114 | CHEM        |     0.99998  |
|  7 | erythrodiol                     |     117 |   127 | CHEM        |     0.999987 |
|  8 | oleanolic acid                  |     133 |   146 | CHEM        |     0.999984 |
|  9 | phenolic acid                   |     153 |   165 | CHEM        |     0.999985 |
| 10 | 4 - hydroxybenzoic acid         |     184 |   206 | CHEM        |     0.999973 |
| 11 | gallic and protocatechuic acids |     209 |   239 | CHEM        |     0.999984 |
| 12 | isocorilagin                    |     245 |   256 | CHEM        |     0.999985 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc4chemd_chemicals_pipeline|
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