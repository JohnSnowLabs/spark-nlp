---
layout: model
title: Pipeline to Detect Living Species (bert_embeddings_bert_base_italian_xxl_cased)
author: John Snow Labs
name: ner_living_species_bert_pipeline
date: 2023-03-13
tags: [it, ner, clinical, licensed, bert]
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

This pretrained pipeline is built on the top of [ner_living_species_bert](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_bert_it_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_pipeline_it_4.3.0_3.2_1678730309748.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_pipeline_it_4.3.0_3.2_1678730309748.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_bert_pipeline", "it", "clinical/models")

text = '''Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_bert_pipeline", "it", "clinical/models")

val text = "Una donna di 74 anni è stata ricoverata con dolore addominale diffuso, ipossia e astenia di 2 settimane di evoluzione. La sua storia personale includeva ipertensione in trattamento con amiloride/idroclorotiazide e dislipidemia controllata con lovastatina. La sua storia familiare era: madre morta di cancro gastrico, fratello con cirrosi epatica di eziologia sconosciuta e sorella con carcinoma epatocellulare. Lo studio eziologico delle diverse cause di malattia epatica cronica comprendeva: virus epatotropi (HBV, HCV) e HIV, studio dell'autoimmunità, ceruloplasmina, ferritina e porfirine nelle urine, tutti risultati negativi. Il paziente è stato messo in trattamento anticoagulante con acenocumarolo e diuretici a tempo indeterminato."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks       |   begin |   end | ner_label   |   confidence |
|---:|:-----------------|--------:|------:|:------------|-------------:|
|  0 | donna            |       4 |     8 | HUMAN       |      0.9997  |
|  1 | personale        |     133 |   141 | HUMAN       |      1       |
|  2 | madre            |     285 |   289 | HUMAN       |      1       |
|  3 | fratello         |     317 |   324 | HUMAN       |      0.9995  |
|  4 | sorella          |     373 |   379 | HUMAN       |      0.9997  |
|  5 | virus epatotropi |     493 |   508 | SPECIES     |      0.75615 |
|  6 | HBV              |     511 |   513 | SPECIES     |      0.9886  |
|  7 | HCV              |     516 |   518 | SPECIES     |      0.9745  |
|  8 | HIV              |     523 |   525 | SPECIES     |      0.9838  |
|  9 | paziente         |     634 |   641 | HUMAN       |      0.9994  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|it|
|Size:|432.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel