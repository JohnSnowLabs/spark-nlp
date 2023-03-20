---
layout: model
title: Pipeline to Detect concepts in drug development trials (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_drug_development_trials_pipeline
date: 2023-03-20
tags: [ner, en, bertfortokenclassification, clinical, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_drug_development_trials](https://nlp.johnsnowlabs.com/2022/06/18/bert_token_classifier_drug_development_trials_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_drug_development_trials_pipeline_en_4.3.0_3.2_1679304929639.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_drug_development_trials_pipeline_en_4.3.0_3.2_1679304929639.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_drug_development_trials_pipeline", "en", "clinical/models")

text = '''In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_drug_development_trials_pipeline", "en", "clinical/models")

val text = "In June 2003, the median overall survival  with and without topotecan were 4.0 and 3.6 months, respectively. The best complete response  ( CR ) , partial response  ( PR ) , stable disease and progressive disease were observed in 23, 63, 55 and 33 patients, respectively, with  topotecan,  and 11, 61, 66 and 32 patients, respectively, without topotecan."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk               |   begin |   end | ner_label     |   confidence |
|---:|:------------------------|--------:|------:|:--------------|-------------:|
|  0 | June 2003               |       3 |    11 | DATE          |     0.996034 |
|  1 | median                  |      18 |    23 | Duration      |     0.999535 |
|  2 | overall survival        |      25 |    40 | End_Point     |     0.996754 |
|  3 | without topotecan       |      52 |    68 | Trial_Group   |     0.976542 |
|  4 | 4.0                     |      75 |    77 | Value         |     0.998101 |
|  5 | 3.6 months              |      83 |    92 | Value         |     0.998159 |
|  6 | complete response  ( CR |     118 |   140 | End_Point     |     0.998629 |
|  7 | partial response  ( PR  |     146 |   167 | End_Point     |     0.998672 |
|  8 | stable disease          |     173 |   186 | End_Point     |     0.996891 |
|  9 | progressive disease     |     192 |   210 | End_Point     |     0.997602 |
| 10 | 23                      |     229 |   230 | Patient_Count |     0.998463 |
| 11 | 63                      |     233 |   234 | Patient_Count |     0.996301 |
| 12 | 55                      |     237 |   238 | Patient_Count |     0.996667 |
| 13 | 33 patients             |     244 |   254 | Patient_Count |     0.995486 |
| 14 | topotecan               |     277 |   285 | Trial_Group   |     0.999624 |
| 15 | 11                      |     293 |   294 | Patient_Count |     0.998747 |
| 16 | 61                      |     297 |   298 | Patient_Count |     0.998314 |
| 17 | 66                      |     301 |   302 | Patient_Count |     0.998066 |
| 18 | 32 patients             |     308 |   318 | Patient_Count |     0.996285 |
| 19 | without topotecan       |     335 |   351 | Trial_Group   |     0.971218 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_drug_development_trials_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel