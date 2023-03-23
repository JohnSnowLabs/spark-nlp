---
layout: model
title: Pipeline to Detect Anatomical Regions (MedicalBertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_anatomy_pipeline
date: 2023-03-20
tags: [anatomy, bertfortokenclassification, ner, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_anatomy](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_anatomy_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_anatomy_pipeline_en_4.3.0_3.2_1679306174114.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_anatomy_pipeline_en_4.3.0_3.2_1679306174114.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_anatomy_pipeline", "en", "clinical/models")

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

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_anatomy_pipeline", "en", "clinical/models")

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
|  0 | great            |     320 |   324 | Multi-tissue_structure |     0.693343 |
|  1 | toe              |     326 |   328 | Multi-tissue_structure |     0.378996 |
|  2 | skin             |     374 |   377 | Organ                  |     0.946453 |
|  3 | conjunctivae     |     554 |   565 | Multi-tissue_structure |     0.929193 |
|  4 | Extraocular      |     574 |   584 | Multi-tissue_structure |     0.858331 |
|  5 | muscles          |     586 |   592 | Organ                  |     0.670788 |
|  6 | Nares            |     613 |   617 | Multi-tissue_structure |     0.573931 |
|  7 | turbinates       |     659 |   668 | Multi-tissue_structure |     0.947797 |
|  8 | Oropharynx       |     683 |   692 | Multi-tissue_structure |     0.458301 |
|  9 | Mucous membranes |     716 |   731 | Tissue                 |     0.811466 |
| 10 | Neck             |     744 |   747 | Organism_subdivision   |     0.879527 |
| 11 | bowel            |     802 |   806 | Organ                  |     0.919502 |
| 12 | great            |     875 |   879 | Multi-tissue_structure |     0.701514 |
| 13 | toe              |     881 |   883 | Multi-tissue_structure |     0.264513 |
| 14 | skin             |     933 |   936 | Organ                  |     0.925361 |
| 15 | toenails         |     943 |   950 | Organism_subdivision   |     0.674937 |
| 16 | foot             |     999 |  1002 | Organism_subdivision   |     0.544587 |
| 17 | great            |    1017 |  1021 | Multi-tissue_structure |     0.818323 |
| 18 | toe              |    1023 |  1025 | Organism_subdivision   |     0.341098 |
| 19 | toenails         |    1031 |  1038 | Organism_subdivision   |     0.75016  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_anatomy_pipeline|
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