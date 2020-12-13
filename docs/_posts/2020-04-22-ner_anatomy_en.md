---
layout: model
title: NerDLModel Anatomy
author: John Snow Labs
name: ner_anatomy_en
date: 2020-04-22
tags: [ner, en, licensed]
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

clinical_ner = NerDLModel.pretrained("ner_anatomy", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[clinical_ner])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

results = model.transform(data)

```

```scala

val ner = NerDLModel.pretrained("ner_anatomy", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(ner))

val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(data)


```

</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a "ner" column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select "token.result" and "ner.result" from your output dataframe or add the "Finisher" to the end of your pipeline.me:

![image](/assets/images/ner_anatomy.png) 

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
Trained on the Anatomical Entity Mention (AnEM) corpus with 'embeddings_clinical'.
http://www.nactem.ac.uk/anatomy/

{:.h2_title}
## Benchmarking
```bash
|    | label                             |   tp |   fp |   fn |     prec |      rec |       f1 |
|---:|:----------------------------------|-----:|-----:|-----:|---------:|---------:|---------:|
|  0 | I-Organism_subdivision            |    0 |    4 |    0 | 0        | 0        | 0        |
|  1 | B-Immaterial_anatomical_entity    |    4 |    0 |    1 | 1        | 0.8      | 0.888889 |
|  2 | B-Cellular_component              |   14 |    4 |    7 | 0.777778 | 0.666667 | 0.717949 |
|  3 | B-Organism_subdivision            |   21 |    7 |    3 | 0.75     | 0.875    | 0.807692 |
|  4 | I-Cell                            |   47 |    8 |    5 | 0.854545 | 0.903846 | 0.878505 |
|  5 | B-Tissue                          |   14 |    2 |   10 | 0.875    | 0.583333 | 0.7      |
|  6 | B-Anatomical_system               |    5 |    1 |    3 | 0.833333 | 0.625    | 0.714286 |
|  7 | B-Organism_substance              |   26 |    2 |    8 | 0.928571 | 0.764706 | 0.83871  |
|  8 | B-Cell                            |   86 |    6 |   11 | 0.934783 | 0.886598 | 0.910053 |
|  9 | I-Organ                           |    1 |    2 |    6 | 0.333333 | 0.142857 | 0.2      |
| 10 | I-Immaterial_anatomical_entity    |    5 |    0 |    0 | 1        | 1        | 1        |
| 11 | I-Tissue                          |   16 |    1 |    6 | 0.941176 | 0.727273 | 0.820513 |
| 12 | I-Pathological_formation          |   20 |    0 |    1 | 1        | 0.952381 | 0.97561  |
| 13 | B-Developing_anatomical_structure |    0 |    0 |    1 | 0        | 0        | 0        |
| 14 | I-Anatomical_system               |    7 |    0 |    0 | 1        | 1        | 1        |
| 15 | I-Developing_anatomical_structure |    0 |    0 |    3 | 0        | 0        | 0        |
| 16 | B-Organ                           |   30 |    7 |    3 | 0.810811 | 0.909091 | 0.857143 |
| 17 | B-Pathological_formation          |   35 |    5 |    3 | 0.875    | 0.921053 | 0.897436 |
| 18 | I-Cellular_component              |    4 |    0 |    3 | 1        | 0.571429 | 0.727273 |
| 19 | I-Multi-tissue_structure          |   26 |   10 |    6 | 0.722222 | 0.8125   | 0.764706 |
| 20 | B-Multi-tissue_structure          |   57 |   23 |    8 | 0.7125   | 0.876923 | 0.786207 |
| 21 | I-Organism_substance              |    6 |    2 |    0 | 0.75     | 1        | 0.857143 |
| 22 | Macro-average                     | 424  |  84  |   88 | 0.731775 | 0.682666 | 0.706368 |
| 23 | Micro-average                     | 424  |  84  |   88 | 0.834646 | 0.828125 | 0.831372 |
```