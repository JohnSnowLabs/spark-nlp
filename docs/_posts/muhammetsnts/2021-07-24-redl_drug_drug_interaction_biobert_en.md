---
layout: model
title: Extract relations between effects of using multiple drugs (ReDL)
author: John Snow Labs
name: redl_drug_drug_interaction_biobert
date: 2021-07-24
tags: [relation_extraction, en, licensed, clinical]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.0.3
spark_version: 2.4
supported: true
annotator: RelationExtractionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Extract potential improvements or harmful effects of Drug-Drug interactions (DDIs) when two or more drugs are taken at the same time or at a certain interval.


## Predicted Entities


`DDI-advise`, `DDI-effect`, `DDI-false`, `DDI-int`, `DDI-mechanism`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_DRUG_DRUG_INT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_drug_drug_interaction_biobert_en_3.0.3_2.4_1627119817997.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_drug_drug_interaction_biobert_en_3.0.3_2.4_1627119817997.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
documenter = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentencer = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentences")

tokenizer = sparknlp.annotators.Tokenizer()\
.setInputCols(["sentences"])\
.setOutputCol("tokens")

pos_tagger = PerceptronModel()\
.pretrained("pos_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"])\
.setOutputCol("pos_tags")

words_embedder = WordEmbeddingsModel() \
.pretrained("embeddings_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"]) \
.setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")\
.setInputCols("sentences", "tokens", "embeddings")\
.setOutputCol("ner_tags") 

ner_converter = NerConverter() \
.setInputCols(["sentences", "tokens", "ner_tags"]) \
.setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
.pretrained("dependency_conllu", "en") \
.setInputCols(["sentences", "pos_tags", "tokens"]) \
.setOutputCol("dependencies")

# Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
.setInputCols(["ner_chunks", "dependencies"])\
.setMaxSyntacticDistance(10)\
.setOutputCol("re_ner_chunks")
#.setRelationPairs(['SYMPTOM-EXTERNAL_BODY_PART_OR_REGION'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
.pretrained('redl_drug_drug_interaction_biobert', 'en', "clinical/models") \
.setPredictionThreshold(0.5)\
.setInputCols(["re_ner_chunks", "sentences"]) \
.setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text="""When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. \
If additional adrenergic drugs are to be administered by any route, \
they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated"""

data = spark.createDataFrame([[text]]).toDF("text")

p_model = pipeline.fit(data)

result = p_model.transform(data)
```
```scala
...
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentencer = new SentenceDetector()
.setInputCols("document")
.setOutputCol("sentences")

val tokenizer = new Tokenizer()
.setInputCols("sentences")
.setOutputCol("tokens")

val pos_tagger = PerceptronModel()
.pretrained("pos_clinical", "en", "clinical/models") 
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel()
.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_posology", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner_tags") 

val ner_converter = new NerConverter()
.setInputCols(Array("sentences", "tokens", "ner_tags"))
.setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
.pretrained("dependency_conllu", "en")
.setInputCols(Array("sentences", "pos_tags", "tokens"))
.setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = RENerChunksFilter()
.setInputCols(Array("ner_chunks", "dependencies"))
.setMaxSyntacticDistance(10)
.setOutputCol("re_ner_chunks")
// .setRelationPairs(Array('SYMPTOM-EXTERNAL_BODY_PART_OR_REGION'))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
.pretrained("redl_drug_drug_interaction_biobert", "en", "clinical/models")
.setPredictionThreshold(0.5)
.setInputCols(Array("re_ner_chunks", "sentences"))
.setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("""When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. If additional adrenergic drugs are to be administered by any route, they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.drug_drug_interaction").predict("""When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. \
If additional adrenergic drugs are to be administered by any route, \
they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated""")
```

</div>


## Results


```bash
|    | relation   | entity1   |   entity1_begin |   entity1_end | chunk1        | entity2   |   entity2_begin |   entity2_end | chunk2       |   confidence |
|---:|:-----------|:----------|----------------:|--------------:|:--------------|:----------|----------------:|--------------:|:-------------|-------------:|
|  0 | DDI-advise | DRUG      |               5 |            17 | carbamazepine | DRUG      |              62 |            73 | aripiprazole |      0.99238 |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|redl_drug_drug_interaction_biobert|
|Compatibility:|Healthcare NLP 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|true|


## Data Source


Trained on DDI Extraction corpus.


## Benchmarking


```bash
Relation           Recall Precision        F1   Support
DDI-advise          0.758     0.874     0.812       211
DDI-effect          0.759     0.754     0.756       348
DDI-false           0.977     0.957     0.967      4097
DDI-int             0.175     0.458     0.253        63
DDI-mechanism       0.783     0.853     0.816       281
Avg.                0.690     0.779     0.721        -
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc3MzUyMTY2M119
-->