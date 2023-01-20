---
layout: model
title: Relation Extraction between anatomical entities and other clinical entities
author: John Snow Labs
name: re_oncology_location_wip
date: 2022-09-27
tags: [licensed, clinical, oncology, en, relation_extraction, anatomy]
task: Relation Extraction
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This relation extraction model links extractions from anatomical entities (such as Site_Breast or Site_Lung) to other clinical entities (such as Tumor_Finding or Cancer_Surgery).

## Predicted Entities

`is_location_of`, `O`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_oncology_location_wip_en_4.0.0_3.0_1664301554732.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_oncology_location_wip_en_4.0.0_3.0_1664301554732.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use relation pairs to include only the combinations of entities that are relevant in your case.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")                

ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
        
pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

re_model = RelationExtractionModel.pretrained("re_oncology_location_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"]) \
    .setOutputCol("relation_extraction") \
    .setRelationPairs(["Tumor_Finding-Site_Breast", "Site_Breast-Tumor_Finding","Tumor_Finding-Anatomical_Site", "Anatomical_Site-Tumor_Finding"]) \
    .setMaxSyntacticDistance(10)
        
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter,
                            pos_tagger,
                            dependency_parser,
                            re_model])

data = spark.createDataFrame([["In April 2011, she first noticed a lump in her right breast."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
    
val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
    .setInputCols(Array("document"))
    .setOutputCol("sentence")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")
    
val word_embeddings = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")                
    
val ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")
    
val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("pos_tags")
    
val dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentence", "pos_tags", "token"))
    .setOutputCol("dependencies")
    
val re_model = RelationExtractionModel.pretrained("re_oncology_location_wip", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunk", "dependencies"))
    .setOutputCol("relation_extraction")
    .setRelationPairs(Array("Tumor_Finding-Site_Breast", "Site_Breast-Tumor_Finding","Tumor_Finding-Anatomical_Site", "Anatomical_Site-Tumor_Finding"))
    .setMaxSyntacticDistance(10)

val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter,
                            pos_tagger,
                            dependency_parser,
                            re_model))

val data = Seq("In April 2011, she first noticed a lump in her right breast.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------+-------------+-------------+-----------+------+---------------+-------------+-----------+------+----------+
|      relation|      entity1|entity1_begin|entity1_end|chunk1|        entity2|entity2_begin|entity2_end|chunk2|confidence|
+--------------+-------------+-------------+-----------+------+---------------+-------------+-----------+------+----------+
|is_location_of|Tumor_Finding|           35|         38|  lump|Anatomical_Site|           53|         58|breast|0.81353307|
+--------------+-------------+-------------+-----------+------+---------------+-------------+-----------+------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_oncology_location_wip|
|Type:|re|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|266.7 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
         label  recall  precision   f1
             O    0.83     0.93   0.88
is_location_of    0.93     0.84   0.88
     macro-avg    0.88     0.88   0.88
```
