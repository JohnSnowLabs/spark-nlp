---
layout: model
title: Pipeline to Detect Living Species (w2v_cc_300d)
author: John Snow Labs
name: ner_living_species_pipeline
date: 2023-03-13
tags: [ca, ner, clinical, licensed]
task: Named Entity Recognition
language: ca
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_living_species](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_ca_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_ca_4.3.0_3.2_1678703245172.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_pipeline_ca_4.3.0_3.2_1678703245172.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_pipeline", "ca", "clinical/models")

text = '''Dona de 47 anys al·lèrgica al iode, fumadora social, intervinguda de varices, dues cesàries i un abscés gluti. Sense altres antecedents mèdics d'interès ni tractament habitual. Viu amb el seu marit i tres fills, treballa com a professora. En el moment de la nostra valoració en la planta de Cirurgia General, la pacient presenta TA 69/40 mm Hg, freqüència cardíaca 120 lpm, taquipnea en repòs, pal·lidesa mucocutánea, mala perfusió distal i afligeix nàusees. L'abdomen és tou, no presenta peritonismo i el dèbit del drenatge abdominal roman sense canvis. Les serologies de Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, citomegalovirus, virus de Epstein Barr, virus varicel·la zoster i parvovirus B19 van ser negatives. No obstant això, es va detectar test de rosa de Bengala positiu per a Brucella, el test de Coombs i les aglutinacions també van ser positives amb un títol 1/40.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_pipeline", "ca", "clinical/models")

val text = "Dona de 47 anys al·lèrgica al iode, fumadora social, intervinguda de varices, dues cesàries i un abscés gluti. Sense altres antecedents mèdics d'interès ni tractament habitual. Viu amb el seu marit i tres fills, treballa com a professora. En el moment de la nostra valoració en la planta de Cirurgia General, la pacient presenta TA 69/40 mm Hg, freqüència cardíaca 120 lpm, taquipnea en repòs, pal·lidesa mucocutánea, mala perfusió distal i afligeix nàusees. L'abdomen és tou, no presenta peritonismo i el dèbit del drenatge abdominal roman sense canvis. Les serologies de Coxiella burnetii, Bartonella henselae, Borrelia burgdorferi, Entamoeba histolytica, Toxoplasma gondii, citomegalovirus, virus de Epstein Barr, virus varicel·la zoster i parvovirus B19 van ser negatives. No obstant això, es va detectar test de rosa de Bengala positiu per a Brucella, el test de Coombs i les aglutinacions també van ser positives amb un títol 1/40."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks              |   begin |   end | ner_label   |   confidence |
|---:|:------------------------|--------:|------:|:------------|-------------:|
|  0 | Dona                    |       0 |     3 | HUMAN       |     1        |
|  1 | marit                   |     192 |   196 | HUMAN       |     0.9867   |
|  2 | fills                   |     205 |   209 | HUMAN       |     0.9822   |
|  3 | professora              |     227 |   236 | HUMAN       |     0.9987   |
|  4 | pacient                 |     312 |   318 | HUMAN       |     0.9986   |
|  5 | Coxiella burnetii       |     573 |   589 | SPECIES     |     0.96365  |
|  6 | Bartonella henselae     |     592 |   610 | SPECIES     |     0.92445  |
|  7 | Borrelia burgdorferi    |     613 |   632 | SPECIES     |     0.91515  |
|  8 | Entamoeba histolytica   |     635 |   655 | SPECIES     |     0.87195  |
|  9 | Toxoplasma gondii       |     658 |   674 | SPECIES     |     0.8935   |
| 10 | citomegalovirus         |     677 |   691 | SPECIES     |     0.9227   |
| 11 | virus de Epstein Barr   |     694 |   714 | SPECIES     |     0.730375 |
| 12 | virus varicel·la zoster |     717 |   739 | SPECIES     |     0.778333 |
| 13 | parvovirus B19          |     743 |   756 | SPECIES     |     0.9138   |
| 14 | Brucella                |     847 |   854 | SPECIES     |     0.9483   |
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
|Language:|ca|
|Size:|1.2 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel