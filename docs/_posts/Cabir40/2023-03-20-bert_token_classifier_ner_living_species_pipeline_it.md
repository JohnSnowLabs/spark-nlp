---
layout: model
title: Pipeline to Detect Living Species
author: John Snow Labs
name: bert_token_classifier_ner_living_species_pipeline
date: 2023-03-20
tags: [it, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: it
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_living_species](https://nlp.johnsnowlabs.com/2022/06/27/bert_token_classifier_ner_living_species_it_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pipeline_it_4.3.0_3.2_1679304618895.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_living_species_pipeline_it_4.3.0_3.2_1679304618895.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_living_species_pipeline", "it", "clinical/models")

text = '''Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_living_species_pipeline", "it", "clinical/models")

val text = "Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk        |   begin |   end | ner_label   |   confidence |
|---:|:-----------------|--------:|------:|:------------|-------------:|
|  0 | donna            |       4 |     8 | HUMAN       |     0.999699 |
|  1 | personale        |     133 |   141 | HUMAN       |     0.999581 |
|  2 | madre            |     285 |   289 | HUMAN       |     0.999633 |
|  3 | fratello         |     317 |   324 | HUMAN       |     0.999641 |
|  4 | sorella          |     373 |   379 | HUMAN       |     0.999622 |
|  5 | virus epatotropi |     493 |   508 | SPECIES     |     0.999333 |
|  6 | HBV              |     511 |   513 | SPECIES     |     0.99968  |
|  7 | HCV              |     516 |   518 | SPECIES     |     0.999616 |
|  8 | HIV              |     523 |   525 | SPECIES     |     0.999383 |
|  9 | paziente         |     634 |   641 | HUMAN       |     0.99977  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_living_species_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|it|
|Size:|410.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel