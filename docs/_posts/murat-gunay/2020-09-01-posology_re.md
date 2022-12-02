---
layout: model
title: Relation Extraction between Posologic entities
author: John Snow Labs
name: posology_re
date: 2020-09-01
task: Relation Extraction
language: en
edition: Healthcare NLP 2.5.5
spark_version: 2.4
tags: [re, en, clinical, licensed, relation extraction]
supported: true
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model extracts relations between posology-related terminology.

## Predicted Entities
`DRUG-DOSAGE`, `DRUG-FREQUENCY`, `DRUG-ADE`, `DRUG-FORM`, `ENDED_BY`, `DRUG-ROUTE`, `DRUG-DURATION`, `DRUG-REASON`, `DRUG-STRENGTH`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_POSOLOGY/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
<button class="button button-orange" disabled>Download</button>

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

tokenizer = Tokenizer() \
    .setInputCols(["sentences"]) \
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
    .pretrained("ner_posology", "en", "clinical/models")\
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
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

light_pipeline = LightPipeline(model)

result = light_pipeline.fullAnnotate("The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.")

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
    .pretrained("ner_posology", "en", "clinical/models")
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
    .pretrained("posology_re")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(4)

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependecy_parser, re_Model))

val data = Seq("The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also given 1 unit of Metformin daily. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
| relation       | entity1  | entity1_begin | entity1_end | chunk1           | entity2   | entity2_begin | entity2_end | chunk2           | confidence |
|----------------|----------|---------------|-------------|------------------|-----------|---------------|-------------|------------------|------------|
| DURATION-DRUG  | DURATION | 493           | 500         | five-day         | DRUG      | 512           | 522         | amoxicillin      | 1.0        |
| DRUG-DURATION  | DRUG     | 681           | 693         | dapagliflozin    | DURATION  | 695           | 708         | for six months   | 1.0        |
| DRUG-ROUTE     | DRUG     | 1940          | 1946        | insulin          | ROUTE     | 1948          | 1951        | drip             | 1.0        |
| DOSAGE-DRUG    | DOSAGE   | 2255          | 2262        | 40 units         | DRUG      | 2267          | 2282        | insulin glargine | 1.0        |
| DRUG-FREQUENCY | DRUG     | 2267          | 2282        | insulin glargine | FREQUENCY | 2284          | 2291        | at night         | 1.0        |
| DOSAGE-DRUG    | DOSAGE   | 2295          | 2302        | 12 units         | DRUG      | 2307          | 2320        | insulin lispro   | 1.0        |
| DRUG-FREQUENCY | DRUG     | 2307          | 2320        | insulin lispro   | FREQUENCY | 2322          | 2331        | with meals       | 1.0        |
| DRUG-STRENGTH  | DRUG     | 2339          | 2347        | metformin        | STRENGTH  | 2349          | 2355        | 1000 mg          | 1.0        |
| DRUG-FREQUENCY | DRUG     | 2339          | 2347        | metformin        | FREQUENCY | 2357          | 2371        | two times a day  | 1.0        |
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|posology_re|
|Compatibility:|Healthcare NLP 2.5.5+|
|Edition:|Official|
|License:|Licensed|
|Language:|[en]|
