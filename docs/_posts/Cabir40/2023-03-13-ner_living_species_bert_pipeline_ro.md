---
layout: model
title: Pipeline to Detect Living Species (bert_base_cased)
author: John Snow Labs
name: ner_living_species_bert_pipeline
date: 2023-03-13
tags: [ro, ner, clinical, licensed, bert]
task: Named Entity Recognition
language: ro
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_living_species_bert](https://nlp.johnsnowlabs.com/2022/06/23/ner_living_species_bert_ro_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_pipeline_ro_4.3.0_3.2_1678729997910.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_living_species_bert_pipeline_ro_4.3.0_3.2_1678729997910.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_living_species_bert_pipeline", "ro", "clinical/models")

text = '''O femeie în vârstă de 26 de ani, însărcinată în 11 săptămâni, a consultat serviciul de urgențe dermatologice pentru că prezenta, de 4 zile, leziuni punctiforme dureroase de debut brusc pe vârful degetelor. Pacientul raportează că leziunile au început pe degete și ulterior s-au extins la degetele de la picioare. Markerii de imunitate, ANA și crioagglutininele, au fost negativi, iar serologia VHB a indicat doar vaccinarea. Pe baza acestor rezultate, diagnosticul de vasculită a fost exclus și, având în vedere diagnosticul suspectat de erupție cutanată cu mănuși și șosete, s-a efectuat serologia pentru virusul Ebstein Barr. Exantemă la mănuși și șosete datorat parvovirozei B19. Având în vedere suspiciunea unei afecțiuni infecțioase cu aceste caracteristici, a fost solicitată serologia pentru EBV, enterovirus și parvovirus B19, cu IgM pozitiv pentru acesta din urmă în două ocazii. De asemenea, nu au existat semne de anemie fetală sau complicații ale acesteia.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_living_species_bert_pipeline", "ro", "clinical/models")

val text = "O femeie în vârstă de 26 de ani, însărcinată în 11 săptămâni, a consultat serviciul de urgențe dermatologice pentru că prezenta, de 4 zile, leziuni punctiforme dureroase de debut brusc pe vârful degetelor. Pacientul raportează că leziunile au început pe degete și ulterior s-au extins la degetele de la picioare. Markerii de imunitate, ANA și crioagglutininele, au fost negativi, iar serologia VHB a indicat doar vaccinarea. Pe baza acestor rezultate, diagnosticul de vasculită a fost exclus și, având în vedere diagnosticul suspectat de erupție cutanată cu mănuși și șosete, s-a efectuat serologia pentru virusul Ebstein Barr. Exantemă la mănuși și șosete datorat parvovirozei B19. Având în vedere suspiciunea unei afecțiuni infecțioase cu aceste caracteristici, a fost solicitată serologia pentru EBV, enterovirus și parvovirus B19, cu IgM pozitiv pentru acesta din urmă în două ocazii. De asemenea, nu au existat semne de anemie fetală sau complicații ale acesteia."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks           |   begin |   end | ner_label   |   confidence |
|---:|:---------------------|--------:|------:|:------------|-------------:|
|  0 | femeie               |       2 |     7 | HUMAN       |     0.9998   |
|  1 | Pacientul            |     206 |   214 | HUMAN       |     0.9993   |
|  2 | VHB                  |     394 |   396 | SPECIES     |     1        |
|  3 | virusul Ebstein Barr |     606 |   625 | SPECIES     |     0.996467 |
|  4 | parvovirozei B19     |     665 |   680 | SPECIES     |     0.5685   |
|  5 | EBV                  |     799 |   801 | SPECIES     |     0.9999   |
|  6 | enterovirus          |     804 |   814 | SPECIES     |     0.9984   |
|  7 | parvovirus B19       |     819 |   832 | SPECIES     |     0.99255  |
|  8 | fetală               |     932 |   937 | HUMAN       |     0.9994   |
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
|Language:|ro|
|Size:|483.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel