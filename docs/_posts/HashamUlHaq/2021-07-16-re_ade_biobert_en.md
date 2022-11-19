---
layout: model
title: Relation extraction between Drugs and ADE (biobert)
author: John Snow Labs
name: re_ade_biobert
date: 2021-07-16
tags: [licensed, en, relation_extraction, clinical]
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


This model is capable of Relating Drugs and adverse reactions caused by them; It predicts if an adverse event is caused by a drug or not. It is based on 'biobert_pubmed_base_cased' embeddings. `1` : Shows the adverse event and drug entities are related, `0` : Shows the adverse event and drug entities are not related.


## Predicted Entities


`0`, `1`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_ade_biobert_en_3.1.2_3.0_1626434941786.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documenter = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

embedder = BertEmbeddings.pretrained("biobert_pubmed_base_cased", "en")\
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel() \
    .pretrained("ner_ade_biobert", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens", "embeddings"]) \
    .setOutputCol("ner_tags")

ner_chunker = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

dependency_parser = sparknlp.annotators.DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

re_model = RelationExtractionModel()\
    .pretrained("re_ade_biobert", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(3)\
    .setPredictionThreshold(0.5)\
    .setRelationPairs(["ade-drug", "drug-ade"])

nlp_pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

text ="""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps"""

annotations = light_pipeline.fullAnnotate(text)
```
```scala
val documenter = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentencer = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val embedder = BertEmbeddings.pretrained("biobert_pubmed_base_cased", "en").pretrained("biobert_pubmed_base_cased", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = MEdicalNerModel()
    .pretrained("ner_ade_biobert", "en", "clinical/models")
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
    .pretrained("re_ade_biobert", "en", 'clinical/models')
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(3) #default: 0 
    .setPredictionThreshold(0.5) #default: 0.5 
    .setRelationPairs(Array("drug-ade", "ade-drug")) # Possible relation pairs. Default: All Relations.

val nlpPipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model))

val data = Seq("""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.ade_biobert").predict("""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")
```

</div>


## Results


```bash
|    | chunk1                        | entitiy1   | chunk2      | entity2 | relation |
|----|-------------------------------|------------|-------------|---------|----------|
| 0  | sever fatigue                 | ADE        | Lipitor     | DRUG    |        1 |
| 1  | cramps                        | ADE        | Lipitor     | DRUG    |        0 |
| 2  | sever fatigue                 | ADE        | voltaren    | DRUG    |        0 |
| 3  | cramps                        | ADE        | voltaren    | DRUG    |        1 |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_ade_biobert|
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
label        precision   recall   f1-score   support
0               0.91      0.92      0.92      1670
1               0.92      0.91      0.91      1673
micro-avg       0.92      0.92      0.92      3343
macro-avg       0.92      0.92      0.92      3343
weighted-avg    0.92      0.92      0.92      3343
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MDg4NjUyMSwtMTQ2MDI5MTQxXX0=
-->