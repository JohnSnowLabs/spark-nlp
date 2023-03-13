---
layout: model
title: Pipeline to Detect Living Species (w2v_cc_300d)
author: John Snow Labs
name: ner_living_species_pipeline
date: 2023-03-13
tags: [fr, ner, clinical, licensed]
task: Named Entity Recognition
language: fr
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_living_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_fr_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_fr_4.3.0_3.2_1678705447222.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_fr_4.3.0_3.2_1678705447222.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_pipeline", "fr", "clinical/models")

text = '''Femme de 47 ans allergique à l'iode, fumeuse sociale, opérée pour des varices, deux césariennes et un abcès fessier. Vit avec son mari et ses trois enfants, travaille comme enseignante. Initialement, le patient a eu une bonne évolution, mais au 2ème jour postopératoire, il a commencé à montrer une instabilité hémodynamique. Les sérologies pour Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, herpès simplex virus 1 et 2, cytomégalovirus, virus d'Epstein Barr, virus de la varicelle et du zona et parvovirus B19 étaient négatives. Cependant, un test au rose Bengale positif pour Brucella, le test de Coombs et les agglutinations étaient également positifs avec un titre de 1/40.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_pipeline", "fr", "clinical/models")

val text = "Femme de 47 ans allergique à l'iode, fumeuse sociale, opérée pour des varices, deux césariennes et un abcès fessier. Vit avec son mari et ses trois enfants, travaille comme enseignante. Initialement, le patient a eu une bonne évolution, mais au 2ème jour postopératoire, il a commencé à montrer une instabilité hémodynamique. Les sérologies pour Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, herpès simplex virus 1 et 2, cytomégalovirus, virus d'Epstein Barr, virus de la varicelle et du zona et parvovirus B19 étaient négatives. Cependant, un test au rose Bengale positif pour Brucella, le test de Coombs et les agglutinations étaient également positifs avec un titre de 1/40."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                       |   begin |   end | ner_label   |   confidence |
|---:|:---------------------------------|--------:|------:|:------------|-------------:|
|  0 | Femme                            |       0 |     4 | HUMAN       |     1        |
|  1 | mari                             |     130 |   133 | HUMAN       |     0.982    |
|  2 | enfants                          |     148 |   154 | HUMAN       |     0.9863   |
|  3 | patient                          |     203 |   209 | HUMAN       |     0.9989   |
|  4 | Coxiella burnetii                |     346 |   362 | SPECIES     |     0.9309   |
|  5 | Bartonella henselae              |     365 |   383 | SPECIES     |     0.99275  |
|  6 | Borrelia burgdorferi             |     386 |   405 | SPECIES     |     0.98795  |
|  7 | Entamoeba histolytica            |     408 |   428 | SPECIES     |     0.98455  |
|  8 | Toxoplasma gondii                |     431 |   447 | SPECIES     |     0.9736   |
|  9 | cytomégalovirus                  |     479 |   493 | SPECIES     |     0.9979   |
| 10 | virus d'Epstein Barr             |     496 |   515 | SPECIES     |     0.788667 |
| 11 | virus de la varicelle et du zona |     518 |   549 | SPECIES     |     0.788543 |
| 12 | parvovirus B19                   |     554 |   567 | SPECIES     |     0.9341   |
| 13 | Brucella                         |     636 |   643 | SPECIES     |     0.9993   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|fr|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel