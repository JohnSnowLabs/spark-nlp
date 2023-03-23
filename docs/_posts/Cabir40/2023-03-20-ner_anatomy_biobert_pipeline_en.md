---
layout: model
title: Pipeline to Detect Anatomical References (biobert)
author: John Snow Labs
name: ner_anatomy_biobert_pipeline
date: 2023-03-20
tags: [ner, clinical, licensed, en]
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

This pretrained pipeline is built on the top of [ner_anatomy_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_anatomy_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_biobert_pipeline_en_4.3.0_3.2_1679312126242.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_biobert_pipeline_en_4.3.0_3.2_1679312126242.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_anatomy_biobert_pipeline", "en", "clinical/models")

text = '''This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.
General: Well-developed female, in no acute distress, afebrile.
HEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.
Neck: No lymphadenopathy.
Chest: Clear.
Abdomen: Positive bowel sounds and soft.
Dermatologic: She has got redness along her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_anatomy_biobert_pipeline", "en", "clinical/models")

val text = "This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.
General: Well-developed female, in no acute distress, afebrile.
HEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.
Neck: No lymphadenopathy.
Chest: Clear.
Abdomen: Positive bowel sounds and soft.
Dermatologic: She has got redness along her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk        |   begin |   end | ner_label              |   confidence |
|---:|:-----------------|--------:|------:|:-----------------------|-------------:|
|  0 | right            |     314 |   318 | Organism_subdivision   |      0.9948  |
|  1 | great            |     320 |   324 | Organism_subdivision   |      0.8723  |
|  2 | toe              |     326 |   328 | Organism_subdivision   |      0.9205  |
|  3 | skin             |     374 |   377 | Organ                  |      1       |
|  4 | Sclerae          |     542 |   548 | Pathological_formation |      0.8029  |
|  5 | Extraocular      |     574 |   584 | Multi-tissue_structure |      0.8437  |
|  6 | muscles          |     586 |   592 | Multi-tissue_structure |      0.8796  |
|  7 | Nares            |     613 |   617 | Organ                  |      0.7716  |
|  8 | turbinates       |     659 |   668 | Multi-tissue_structure |      0.9257  |
|  9 | Mucous membranes |     716 |   731 | Cell                   |      0.70435 |
| 10 | Neck             |     744 |   747 | Organism_subdivision   |      0.9982  |
| 11 | Abdomen          |     784 |   790 | Organism_subdivision   |      0.8902  |
| 12 | bowel            |     802 |   806 | Organism_subdivision   |      1       |
| 13 | right            |     869 |   873 | Organism_subdivision   |      0.9967  |
| 14 | toe              |     881 |   883 | Organism_subdivision   |      0.9816  |
| 15 | skin             |     933 |   936 | Organ                  |      1       |
| 16 | toenails         |     943 |   950 | Organism_subdivision   |      0.9999  |
| 17 | foot             |     999 |  1002 | Organism_subdivision   |      0.9831  |
| 18 | toe              |    1023 |  1025 | Organism_subdivision   |      0.9653  |
| 19 | toenails         |    1031 |  1038 | Organism_subdivision   |      0.9999  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.2 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel