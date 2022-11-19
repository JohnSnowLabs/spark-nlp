---
layout: model
title: Detect Anatomical Regions (MedicalBertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_anatomy
date: 2022-01-06
tags: [anatomy, bertfortokenclassification, ner, en, licensed]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: MedicalBertForTokenClassifier
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Pretrained named entity recognition deep learning model for anatomy terms. This model is trained with the BertForTokenClassification method from the transformers library and imported into Spark NLP.


## Predicted Entities


`Anatomical_system`, `Cell`, `Cellular_component`, `Developing_anatomical_structure`, `Immaterial_anatomical_entity`, `Multi-tissue_structure`, `Organ`, `Organism_subdivision`, `Organism_substance`, `Pathological_formation`, `Tissue`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_anatomy_en_3.3.4_2.4_1641454747169.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols(["document"])\
.setOutputCol("token")

tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_anatomy", "en", "clinical/models")\
.setInputCols("token", "document")\
.setOutputCol("ner")\
.setCaseSensitive(True)

ner_converter = NerConverter() \
.setInputCols(["document","token","ner"]) \
.setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[documentAssembler, tokenizer, tokenClassifier, ner_converter])

data = spark.createDataFrame([["""This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.\nGeneral: Well-developed female, in no acute distress, afebrile.\nHEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.\nNeck: No lymphadenopathy.\nChest: Clear.\nAbdomen: Positive bowel sounds and soft.\nDermatologic: She has got redness along her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short."""]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = new Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_anatomy", "en", "clinical/models")
.setInputCols(Array("document","token"))
.setOutputCol("ner")
.setCaseSensitive(True)

val ner_converter = new NerConverter()
.setInputCols(Array("document","token","ner"))
.setOutputCol("ner_chunk")

val pipeline =  new Pipeline().setStages(Array(documentAssembler, tokenizer, tokenClassifier, ner_converter))

val data = Seq("""This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.\nGeneral: Well-developed female, in no acute distress, afebrile.\nHEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.\nNeck: No lymphadenopathy.\nChest: Clear.\nAbdomen: Positive bowel sounds and soft.\nDermatologic: She has got redness along her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.token_bert.ner_anatomy").predict("""This is an 11-year-old female who comes in for two different things. 1. She was seen by the allergist. No allergies present, so she stopped her Allegra, but she is still real congested and does a lot of snorting. They do not notice a lot of snoring at night though, but she seems to be always like that. 2. On her right great toe, she has got some redness and erythema. Her skin is kind of peeling a little bit, but it has been like that for about a week and a half now.\nGeneral: Well-developed female, in no acute distress, afebrile.\nHEENT: Sclerae and conjunctivae clear. Extraocular muscles intact. TMs clear. Nares patent. A little bit of swelling of the turbinates on the left. Oropharynx is essentially clear. Mucous membranes are moist.\nNeck: No lymphadenopathy.\nChest: Clear.\nAbdomen: Positive bowel sounds and soft.\nDermatologic: She has got redness along her right great toe, but no bleeding or oozing. Some dryness of her skin. Her toenails themselves are very short and even on her left foot and her left great toe the toenails are very short.""")
```

</div>


## Results


```bash
+-------------------+----------------------+
|chunk              |ner_label             |
+-------------------+----------------------+
|great toe          |Multi-tissue_structure|
|skin               |Organ                 |
|conjunctivae       |Multi-tissue_structure|
|Extraocular muscles|Multi-tissue_structure|
|Nares              |Multi-tissue_structure|
|turbinates         |Multi-tissue_structure|
|Oropharynx         |Multi-tissue_structure|
|Mucous membranes   |Tissue                |
|Neck               |Organism_subdivision  |
|bowel              |Organ                 |
|great toe          |Multi-tissue_structure|
|skin               |Organ                 |
|toenails           |Organism_subdivision  |
|foot               |Organism_subdivision  |
|great toe          |Multi-tissue_structure|
|toenails           |Organism_subdivision  |
+-------------------+----------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_anatomy|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|404.4 MB|
|Case sensitive:|true|
|Max sentense length:|512|


## Data Source


Trained on the Anatomical Entity Mention (AnEM) corpus with 'embeddings_clinical'. http://www.nactem.ac.uk/anatomy/


## Benchmarking


```bash
label  precision    recall  f1-score   support
B-Anatomical_system       1.00      0.50      0.67         4
B-Cell       0.89      0.96      0.92        74
B-Cellular_component       0.97      0.81      0.88        36
B-Developing_anatomical_structure       1.00      0.50      0.67         6
B-Immaterial_anatomical_entity       0.60      0.75      0.67         4
B-Multi-tissue_structure       0.75      0.86      0.80        58
B-Organ       0.86      0.88      0.87        48
B-Organism_subdivision       0.62      0.42      0.50        12
B-Organism_substance       0.89      0.81      0.85        31
B-Pathological_formation       0.91      0.91      0.91        32
B-Tissue       0.94      0.76      0.84        21
I-Anatomical_system       1.00      1.00      1.00         1
I-Cell       1.00      0.84      0.91        62
I-Cellular_component       0.92      0.85      0.88        13
I-Developing_anatomical_structure       1.00      1.00      1.00         1
I-Immaterial_anatomical_entity       1.00      1.00      1.00         1
I-Multi-tissue_structure       1.00      0.77      0.87        26
I-Organ       0.80      0.80      0.80         5
I-Organism_substance       1.00      0.71      0.83         7
I-Pathological_formation       1.00      0.94      0.97        16
I-Tissue       0.93      0.93      0.93        15
accuracy         -         -       0.84       473
macro-avg       0.87      0.77      0.83       473
weighted-avg       0.90      0.84      0.86       473
```
