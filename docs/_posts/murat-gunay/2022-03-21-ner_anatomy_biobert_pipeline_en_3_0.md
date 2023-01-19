---
layout: model
title: Pipeline to Detect Anatomical References (biobert)
author: John Snow Labs
name: ner_anatomy_biobert_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_biobert_pipeline_en_3.4.1_3.0_1647873806641.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_biobert_pipeline_en_3.4.1_3.0_1647873806641.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_anatomy_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.\nGeneral: Well-developed female, in no acute distress, afebrile.\nHEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.\nNeck: No lymphadenopathy.\nChest: Clear.\nAbdomen: Positive bowel sounds and soft.\nDermatologic: She has got redness along the lateral portion of her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.")
```
```scala
val pipeline = new PretrainedPipeline("ner_anatomy_biobert_pipeline", "en", "clinical/models")


pipeline.annotate("This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.\nGeneral: Well-developed female, in no acute distress, afebrile.\nHEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.\nNeck: No lymphadenopathy.\nChest: Clear.\nAbdomen: Positive bowel sounds and soft.\nDermatologic: She has got redness along the lateral portion of her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.")
```
</div>

## Results

```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|right              |Organism_subdivision  |
|great              |Organism_subdivision  |
|toe                |Organism_subdivision  |
|skin               |Organ                 |
|Sclerae            |Pathological_formation|
|Extraocular muscles|Multi-tissue_structure|
|Nares              |Organ                 |
|turbinates         |Multi-tissue_structure|
|Mucous membranes   |Cell                  |
|Abdomen            |Organism_subdivision  |
|bowel              |Organism_subdivision  |
|right              |Organism_subdivision  |
|toe                |Organism_subdivision  |
|skin               |Organ                 |
|toenails           |Organism_subdivision  |
|foot               |Organism_subdivision  |
|toe                |Organism_subdivision  |
|toenails           |Organism_subdivision  |
+-------------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
