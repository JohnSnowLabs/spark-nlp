---
layout: model
title: Relation Extraction between different oncological entity types (granular version)
author: John Snow Labs
name: re_oncology_granular_wip
date: 2022-09-27
tags: [licensed, clinical, oncology, en, relation_extraction, temporal, test, biomarker, anatomy]
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

Using this relation extraction model, four relation types can be identified: is_date_of (between date entities and other clinical entities), is_size_of (between Tumor_Finding and Tumor_Size), is_location_of (between anatomical entities and other entities) and is_finding_of (between test entities and their results).

## Predicted Entities

`is_size_of`, `is_finding_of`, `is_date_of`, `is_location_of`, `O`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_oncology_granular_wip_en_4.0.0_3.0_1664301874672.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_oncology_granular_wip_en_4.0.0_3.0_1664301874672.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use realation pairs to include only the combinations of entities that are relevant in your case.

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

re_model = RelationExtractionModel.pretrained("re_oncology_granular_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"]) \
    .setOutputCol("relation_extraction") \
    .setRelationPairs(["Tumor_Finding-Tumor_Size", "Tumor_Size-Tumor_Finding", "Cancer_Surgery-Relative_Date", "Relative_Date-Cancer_Surgery"]) \
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

data = spark.createDataFrame([["A mastectomy was performed two months ago, and a 3 cm mass was extracted."]]).toDF("text")

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
    
val re_model = RelationExtractionModel.pretrained("re_oncology_granular_wip", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunk", "dependencies"))
    .setOutputCol("relation_extraction")
    .setRelationPairs(Array("Tumor_Finding-Tumor_Size", "Tumor_Size-Tumor_Finding", "Cancer_Surgery-Relative_Date", "Relative_Date-Cancer_Surgery"))
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

val data = Seq("A mastectomy was performed two months ago, and a 3 cm mass was extracted.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
    chunk1        entity1         chunk2       entity2   relation confidence
mastectomy Cancer_Surgery two months ago Relative_Date is_date_of 0.91336143
      3 cm     Tumor_Size           mass Tumor_Finding is_size_of 0.96745735
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_oncology_granular_wip|
|Type:|re|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|267.2 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
      relation  recall  precision   f1
    is_size_of    0.96       0.73 0.83
             O    0.67       0.93 0.78
 is_finding_of    0.94       0.75 0.83
    is_date_of    0.94       0.54 0.69
is_location_of    0.94       0.81 0.87
     macro-avg    0.89       0.75 0.80
```
