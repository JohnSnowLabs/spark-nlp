---
layout: model
title: Relation Extraction between generic entities
author: John Snow Labs
name: posology_re
date: 2022-12-20
task: Relation Extraction
language: en
edition: Healthcare NLP 4.2.3
spark_version: 3.0
tags: [re, en, clinical, licensed, relation extraction, generic]
supported: true
annotator: RelationExtractionModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
We already have more than 80 relation extraction (RE) models that can extract relations between certain named entities. Nevertheless, there are some rare entities or cases that you may not find the right RE or the one you find may not work as expected due to nature of your dataset. In order to ease this burden, we are releasing a generic RE model (generic_re) that can be used between any named entities using the syntactic distances, POS tags and dependency tree between the entities. You can tune this model by using the setMaxSyntacticDistance param.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_POSOLOGY/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

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

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel()\
    .pretrained("ner_oncology_wip", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")    

ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

reModel = RelationExtractionModel()\
    .pretrained("generic_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene", "Pathology_Test-Pathology_Result", "Pathology_Result-Pathology_Test"]) \
    .setMaxSyntacticDistance(4)

pipeline = Pipeline(stages=[
    documenter,
    sentencer,
    tokenizer, 
    words_embedder, 
    pos_tagger, 
    ner_tagger,
    ner_chunker,
    dependency_parser,
    reModel
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

text = """Pathology showed tumor cells, which were positive for estrogen and progesterone receptors."""
```

```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

sentence_detector = SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

tokenizer = Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val ner_tagger = MedicalNerModel()
    .pretrained("ner_oncology_wip", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")    

val ner_chunker = new NerConverterInternal()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val re_Model = RelationExtractionModel()
    .pretrained("generic_re")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setRelationPairs(["Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene", "Pathology_Test-Pathology_Result", "Pathology_Result-Pathology_Test"]) \
    .setMaxSyntacticDistance(4)

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependecy_parser, re_Model))

val data = Seq("Pathology showed tumor cells, which were positive for estrogen and progesterone receptors.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
|sentence |entity1_begin |entity1_end | chunk1    | entity1          |entity2_begin |entity2_end | chunk2                 | entity2          | relation                        |confidence|
|--------:|-------------:|-----------:|:----------|:-----------------|-------------:|-----------:|:-----------------------|:-----------------|:--------------------------------|----------|
|       0 |            1 |          9 | Pathology | Pathology_Test   |           18 |         28 | tumor cells            | Pathology_Result | Pathology_Test-Pathology_Result |         1|
|       0 |           42 |         49 | positive  | Biomarker_Result |           55 |         62 | estrogen               | Biomarker        | Biomarker_Result-Biomarker      |         1|
|       0 |           42 |         49 | positive  | Biomarker_Result |           68 |         89 | progesterone receptors | Biomarker        | Biomarker_Result-Biomarker      |         1|
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|generic_re|
|Compatibility:|Healthcare NLP 4.2.3+|
|Edition:|Official|
|License:|Licensed|
|Language:|[en]|
