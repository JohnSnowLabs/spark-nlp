---
layout: model
title: Relation extraction between Drugs and ADE (ReDL)
author: John Snow Labs
name: redl_ade_biobert
date: 2021-07-12
tags: [relation_extraction, en, clinical, licensed, ade, biobert]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.1.2
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is an end-to-end trained BioBERT model, capable of Relating Drugs and adverse reactions caused by them; It predicts if an adverse event is caused by a drug or not. `1` : Shows the adverse event and drug entities are related, `0` : Shows the adverse event and drug entities are not related.


## Predicted Entities


`0`, `1`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ADE/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/RE_ADE.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_ade_biobert_en_3.1.2_3.0_1626105541347.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_ade_biobert_en_3.1.2_3.0_1626105541347.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


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

tokenizer = sparknlp.annotators.Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel() \
    .pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

ner_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

dependency_parser = sparknlp.annotators.DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

# Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setMaxSyntacticDistance(10)\
    .setOutputCol("re_ner_chunks")\
    .setRelationPairs(['ade-drug', 'drug-ade'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
    .pretrained('redl_ade_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter,
                            sentencer,
                            tokenizer, 
                            pos_tagger,
                            words_embedder, 
                            ner_tagger, 
                            ner_converter, 
                            dependency_parser, 
                            re_ner_chunk_filter, 
                            re_model])

light_pipeline = LightPipeline(pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

text ="""Been taking Lipitor for 15 years , have experienced severe fatigue a lot. The doctor moved me to voltarene 2 months ago, so far I have only had muscle cramps."""

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

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models")
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

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setMaxSyntacticDistance(10)
    .setOutputCol("re_ner_chunks")
    .setRelationPairs(Array("drug-ade", "ade-drug"))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
    .pretrained("redl_ade_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, 
                                              sentencer, 
                                              tokenizer, 
                                              words_embedder, 
                                              ner_tagger, 
                                              ner_converter, 
                                              pos_tagger, 
                                              dependency_parser, 
                                              re_ner_chunk_filter, 
                                              re_model))

val data = Seq("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot. The doctor moved me to voltarene 2 months ago, so far I have only had muscle cramps.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.adverse_drug_events.clinical.biobert").predict("""Been taking Lipitor for 15 years , have experienced severe fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")
```

</div>


## Results


```bash
| relation | entity1 | entity1_begin | entity1_end | chunk1    | entity2 | entity2_begin | entity2_end | chunk2         | confidence |
|---------:|:--------|--------------:|------------:|:----------|:--------|--------------:|------------:|:---------------|-----------:|
|        1 | DRUG    |            12 |          18 | Lipitor   | ADE     |            52 |          65 | severe fatigue |   0.999317 |
|        0 | DRUG    |            97 |         105 | voltarene | ADE     |           144 |         156 | muscle cramps  |   0.774904 |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|redl_ade_biobert|
|Compatibility:|Healthcare NLP 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[redl_ner_chunks, document]|
|Output Labels:|[relations]|
|Language:|en|


## Data Source


This model is trained on custom data annotated by JSL.


## Benchmarking


```bash
label              Recall Precision        F1   Support
0                   0.829     0.895     0.861      1146
1                   0.955     0.923     0.939      2454
Avg.                0.892     0.909     0.900      -
Weighted-Avg.       0.915     0.914     0.914      -
```
