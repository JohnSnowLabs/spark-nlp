---
layout: model
title: Relation Extraction between dates and other entities
author: John Snow Labs
name: re_oncology_temporal_wip
date: 2022-09-27
tags: [licensed, clinical, oncology, en, relation_extraction, temporal]
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

This relation extraction model links Date and Relative_Date extractions to clinical entities such as Test or Cancer_Dx.

## Predicted Entities

`is_date_of`, `O`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_oncology_temporal_wip_en_4.0.0_3.0_1664297421226.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_oncology_temporal_wip_en_4.0.0_3.0_1664297421226.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Each relevant relation pair in the pipeline should include one date entity (Date or Relative_Date) and a clinical entity (such as Pathology_Test, Cancer_Dx or Chemotherapy).

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

re_model = RelationExtractionModel.pretrained("re_oncology_temporal_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"]) \
    .setOutputCol("relation_extraction") \
    .setRelationPairs(["Cancer_Dx-Date", "Date-Cancer_Dx", "Relative_Date-Cancer_Dx", "Cancer_Dx-Relative_Date", "Cancer_Surgery-Date", "Date-Cancer_Surgery", "Cancer_Surger-Relative_Date", "Relative_Date-Cancer_Surgery"]) \
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

data = spark.createDataFrame([["Her breast cancer was diagnosed three years ago, and a bilateral mastectomy was performed last month."]]).toDF("text")

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
    
val re_model = RelationExtractionModel.pretrained("re_oncology_temporal_wip", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunk", "dependencies"))
    .setOutputCol("relation_extraction")
    .setRelationPairs(Array("Cancer_Dx-Date", "Date-Cancer_Dx", "Relative_Date-Cancer_Dx", "Cancer_Dx-Relative_Date", "Cancer_Surgery-Date", "Date-Cancer_Surgery", "Cancer_Surger-Relative_Date", "Relative_Date-Cancer_Surgery"))
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

val data = Seq("Her breast cancer was diagnosed three years ago, and a bilateral mastectomy was performed last month.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

## Results

```bash
         chunk1          entity1            chunk2         entity2    relation   confidence
  breast cancer        Cancer_Dx   three years ago   Relative_Date  is_date_of    0.5886298
  breast cancer        Cancer_Dx        last month   Relative_Date           O    0.9708738
three years ago    Relative_Date        mastectomy  Cancer_Surgery           O    0.6020852
     mastectomy   Cancer_Surgery        last month   Relative_Date  is_date_of    0.9277692
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_oncology_temporal_wip|
|Type:|re|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|265.6 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
     label  recall  precision     f1
         O    0.79       0.76   0.77
is_date_of    0.74       0.77   0.75
 macro-avg    0.76       0.76   0.76
```
