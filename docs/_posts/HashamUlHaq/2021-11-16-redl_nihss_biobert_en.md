---
layout: model
title: Extract relations between NIHSS entities
author: John Snow Labs
name: redl_nihss_biobert
date: 2021-11-16
tags: [en, licensed, clinical, relation_extraction]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.3.2
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Relate scale items and their measurements according to NIHSS guidelines.


## Predicted Entities


`Has_Value` : Measurement is related to the entity, `0` : Measurement is not related to the entity


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_NIHSS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_nihss_biobert_en_3.3.2_3.0_1637038860417.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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

pos_tagger = PerceptronModel().pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

words_embedder = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"]) \
    .setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_nihss", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

ner_converter = NerConverter() \
    .setInputCols(["sentences", "tokens", "ner_tags"]) \
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en") \
    .setInputCols(["sentences", "pos_tags", "tokens"]) \
    .setOutputCol("dependencies")

# Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunks", "dependencies"])\
    .setMaxSyntacticDistance(10)\
    .setOutputCol("re_ner_chunks")

re_model = RelationExtractionDLModel().pretrained('redl_nihss_biobert', 'en', "clinical/models") \
    .setPredictionThreshold(0.5)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text= "There , her initial NIHSS score was 4 , as recorded by the ED physicians . This included 2 for weakness in her left leg and 2 for what they felt was subtle ataxia in her left arm and leg ."

p_model = pipeline.fit(spark.createDataFrame([[text]]).toDF("text"))

result = p_model.transform(spark.createDataFrame(pd.DataFrame({'text': [text]})))
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

val pos_tagger = PerceptronModel().pretrained("pos_clinical", "en", "clinical/models") 
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_nihss", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags") 

val ner_converter = new NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols(Array("ner_chunks", "dependencies"))
    .setMaxSyntacticDistance(10)
    .setOutputCol("re_ner_chunks")

val re_model = RelationExtractionDLModel().pretrained("redl_nihss_biobert", "en", "clinical/models")
    .setPredictionThreshold(0.5)
    .setInputCols(Array("re_ner_chunks", "sentences"))
    .setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("""There , her initial NIHSS score was 4 , as recorded by the ED physicians . This included 2 for weakness in her left leg and 2 for what they felt was subtle ataxia in her left arm and leg .""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.extract_relation.nihss").predict("""There , her initial NIHSS score was 4 , as recorded by the ED physicians . This included 2 for weakness in her left leg and 2 for what they felt was subtle ataxia in her left arm and leg .""")
```

</div>


## Results


```bash
| chunk1                                | entity1      |   entity1_begin |   entity1_end | entity2     |   chunk2 |   entity2_begin |   entity2_end | relation   |
|:--------------------------------------|:-------------|----------------:|--------------:|:------------|---------:|----------------:|--------------:|:-----------|
| initial NIHSS score                   | NIHSS        |              12 |            30 | Measurement |        4 |              36 |            36 | Has_Value  |
| left leg                              | 6a_LeftLeg   |             111 |           118 | Measurement |        2 |              89 |            89 | Has_Value  |
| subtle ataxia in her left arm and leg | 7_LimbAtaxia |             149 |           185 | Measurement |        2 |             124 |           124 | Has_Value  |
| left leg                              | 6a_LeftLeg   |             111 |           118 | Measurement |        4 |              36 |            36 | 0          |
| initial NIHSS score                   | NIHSS        |              12 |            30 | Measurement |        2 |             124 |           124 | 0          |
| subtle ataxia in her left arm and leg | 7_LimbAtaxia |             149 |           185 | Measurement |        4 |              36 |            36 | 0          |
| subtle ataxia in her left arm and leg | 7_LimbAtaxia |             149 |           185 | Measurement |        2 |              89 |            89 | 0          |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|redl_nihss_biobert|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|


## Data Source


@article{wangnational,
title={National Institutes of Health Stroke Scale (NIHSS) Annotations for the MIMIC-III Database},
author={Wang, Jiayang and Huang, Xiaoshuo and Yang, Lin and Li, Jiao}
}


## Benchmarking


```bash
Relation           Recall Precision        F1   Support
0                   0.989     0.976     0.982       611
Has_Value           0.983     0.992     0.988       889
Avg.                0.986     0.984     0.985		-
Weighted-Avg.       0.985     0.985     0.985		-
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQwMDIzMTM2Ml19
-->