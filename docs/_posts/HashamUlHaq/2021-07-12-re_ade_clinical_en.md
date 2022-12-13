---
layout: model
title: Relation extraction between Drugs and ADE
author: John Snow Labs
name: re_ade_clinical
date: 2021-07-12
tags: [licensed, clinical, en, relation_extraction, ade]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.1.2
spark_version: 3.0
supported: true
annotator: RelationExtractionModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is capable of Relating Drugs and adverse reactions caused by them; It predicts if an adverse event is caused by a drug or not. `1` : Shows the adverse event and drug entities are related, `0` : Shows the adverse event and drug entities are not related.


## Predicted Entities


`0`, `1`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_ADE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_ade_clinical_en_3.1.2_3.0_1626104637779.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_ade_clinical_en_3.1.2_3.0_1626104637779.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use


In the table below, `re_ade_clinical` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.


|     RE MODEL    | RE MODEL LABES |     NER MODEL    | RE PAIRS                     |
|:---------------:|:--------------:|:----------------:|------------------------------|
| re_ade_clinical |     0<br>1     | ner_ade_clinical | ["ade-drug",<br> "drug-ade"] |




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel() \
    .pretrained("ner_ade_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner_tags"]) \
    .setOutputCol("ner_chunks")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

dependency_parser = sparknlp.annotators.DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos_tags", "token"])\
    .setOutputCol("dependencies")

re_model = RelationExtractionModel()\
    .pretrained("re_ade_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(10)\
    .setPredictionThreshold(0.1)\
    .setRelationPairs(["ade-drug", "drug-ade"])\
    .setRelationPairsCaseSensitive(False) 

nlp_pipeline = Pipeline(stages=[documentAssembler,
                                sentenceDetector,  
                                tokenizer, 
                                words_embedder, 
                                ner_tagger, 
                                ner_converter,
                                pos_tagger, 
                                dependency_parser, 
                                re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

text ="""Been taking Lipitor for 15 years , have experienced severe fatigue a lot. The doctor moved me to voltarene 2 months ago, so far I have only had muscle cramps. """

annotations = light_pipeline.fullAnnotate(text)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
         
val sentence_detector = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = NerDLModel()
    .pretrained("ner_ade_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val re_model = RelationExtractionModel()
    .pretrained("re_ade_clinical", "en", 'clinical/models')
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(3) 
    .setPredictionThreshold(0.5) 
    .setRelationPairs(Array("drug-ade", "ade-drug"))

val nlpPipeline = new Pipeline().setStages(Array(
                                documentAssembler,
                                sentenceDetector,  
                                tokenizer, 
                                words_embedder, 
                                ner_tagger, 
                                ner_converter,
                                pos_tagger, 
                                dependency_parser, 
                                re_model))

val data = Seq("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot. The doctor moved me to voltarene 2 months ago, so far I have only had muscle cramps. """).toDS.toDF("text")

val result = nlpPipeline.fit(data).transform(data)

```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.adverse_drug_events.clinical").predict("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot. The doctor moved me to voltarene 2 months ago, so far I have only had muscle cramps.""")
```

</div>


## Results


```bash
| relation | entity1 | entity1_begin | entity1_end | chunk1    | entity2 | entity2_begin | entity2_end | chunk2         | confidence |
|---------:|:--------|--------------:|------------:|:----------|:--------|--------------:|------------:|:---------------|-----------:|
|        1 | DRUG    |            12 |          18 | Lipitor   | ADE     |            52 |          65 | severe fatigue |   1        |
|        1 | DRUG    |            97 |         105 | voltarene | ADE     |           144 |         156 | muscle cramps  |   0.997283 |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_ade_clinical|
|Type:|re|
|Compatibility:|Healthcare NLP 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|


## Data Source


This model is trained on custom data annotated by JSL.


## Benchmarking


```bash
label      precision    recall  f1-score   support
0               0.86      0.88      0.87      1787
1               0.92      0.90      0.91      2586
micro-avg       0.89      0.89      0.89      4373
macro-avg       0.89      0.89      0.89      4373
weighted-avg    0.89      0.89      0.89      4373
```