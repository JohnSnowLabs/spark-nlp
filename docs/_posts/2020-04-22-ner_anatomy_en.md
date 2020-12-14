---
layout: model
title: Detect Anatomical Regions
author: John Snow Labs
name: ner_anatomy_en
date: 2020-04-22
tags: [ner, en, clinical, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---
 
## Description

Pretrained named entity recognition deep learning model for anatomy terms. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN. 

## Predicted Entities 
Anatomical_system, Cell, Cellular_component, Developing_anatomical_structure, Immaterial_anatomical_entity, Multi-tissue_structure, Organ, Organism_subdivision, Organism_substance, Pathological_formation, Tissue

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_ANATOMY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_anatomy_en_2.4.2_2.4_1587513307751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
clinical_ner = NerDLModel.pretrained("ner_anatomy", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

...

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": [
    """This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.
General: Well-developed female, in no acute distress, afebrile.
HEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.
Neck: No lymphadenopathy.
Chest: Clear.
Abdomen: Positive bowel sounds and soft.
Dermatologic: She has got redness along the lateral portion of her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short."""
]})))

```

```scala
...

val ner = NerDLModel.pretrained("ner_anatomy", "en", "clinical/models")
  .setInputCols("sentence", "token", "embeddings")
  .setOutputCol("ner")

...

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, ner, ner_converter))

val result = pipeline.fit(Seq.empty["""This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.
General: Well-developed female, in no acute distress, afebrile.
HEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist. Neck: No lymphadenopathy. Chest: Clear. Abdomen: Positive bowel sounds and soft. Dermatologic: She has got redness along the lateral portion of her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short."""].toDS.toDF("text")).transform(data)

```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a ``"ner"`` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select ``"token.result"`` and ``"ner.result"`` from your output dataframe or add the ``"Finisher"`` to the end of your pipeline.

```bash
+-------------------+----------------------+
|chunk              |ner                   |
+-------------------+----------------------+
|skin               |Organ                 |
|Extraocular muscles|Organ                 |
|turbinates         |Multi-tissue_structure|
|Mucous membranes   |Tissue                |
|Neck               |Organism_subdivision  |
|bowel              |Organ                 |
|skin               |Organ                 |
+-------------------+----------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_anatomy_en_2.4.2_2.4|
|Type:|ner|
|Compatibility:|Spark NLP 2.4.2|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence,token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Data Source
Trained on the Anatomical Entity Mention (AnEM) corpus with ``'embeddings_clinical'``.
http://www.nactem.ac.uk/anatomy/

{:.h2_title}
## Benchmarking
```bash
|    | label                             |   tp |   fp |   fn |     prec |      rec |       f1 |
|---:|----------------------------------:|-----:|-----:|-----:|---------:|---------:|---------:|
|  0 | B-Immaterial_anatomical_entity    |    4 |    0 |    1 | 1        | 0.8      | 0.888889 |
|  1 | B-Cellular_component              |   14 |    4 |    7 | 0.777778 | 0.666667 | 0.717949 |
|  2 | B-Organism_subdivision            |   21 |    7 |    3 | 0.75     | 0.875    | 0.807692 |
|  3 | I-Cell                            |   47 |    8 |    5 | 0.854545 | 0.903846 | 0.878505 |
|  4 | B-Tissue                          |   14 |    2 |   10 | 0.875    | 0.583333 | 0.7      |
|  5 | B-Anatomical_system               |    5 |    1 |    3 | 0.833333 | 0.625    | 0.714286 |
|  6 | B-Organism_substance              |   26 |    2 |    8 | 0.928571 | 0.764706 | 0.83871  |
|  7 | B-Cell                            |   86 |    6 |   11 | 0.934783 | 0.886598 | 0.910053 |
|  8 | I-Immaterial_anatomical_entity    |    5 |    0 |    0 | 1        | 1        | 1        |
|  9 | I-Tissue                          |   16 |    1 |    6 | 0.941176 | 0.727273 | 0.820513 |
| 10 | I-Pathological_formation          |   20 |    0 |    1 | 1        | 0.952381 | 0.97561  |
| 11 | I-Anatomical_system               |    7 |    0 |    0 | 1        | 1        | 1        |
| 12 | B-Organ                           |   30 |    7 |    3 | 0.810811 | 0.909091 | 0.857143 |
| 13 | B-Pathological_formation          |   35 |    5 |    3 | 0.875    | 0.921053 | 0.897436 |
| 14 | I-Cellular_component              |    4 |    0 |    3 | 1        | 0.571429 | 0.727273 |
| 15 | I-Multi-tissue_structure          |   26 |   10 |    6 | 0.722222 | 0.8125   | 0.764706 |
| 16 | B-Multi-tissue_structure          |   57 |   23 |    8 | 0.7125   | 0.876923 | 0.786207 |
| 17 | I-Organism_substance              |    6 |    2 |    0 | 0.75     | 1        | 0.857143 |
| 18 | Macro-average                     | 424  |  84  |   88 | 0.731775 | 0.682666 | 0.706368 |
| 19 | Micro-average                     | 424  |  84  |   88 | 0.834646 | 0.828125 | 0.831372 |
```