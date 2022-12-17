---
layout: model
title: Relation Extraction between Test and Results
author: John Snow Labs
name: re_oncology_test_result_wip
date: 2022-09-27
tags: [licensed, clinical, oncology, en, relation_extraction, test]
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

This relation extraction model links test extractions to their corresponding results.

## Predicted Entities

`is_finding_of`, `O`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_oncology_test_result_wip_en_4.0.0_3.0_1664301319746.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Each relevant relation pair in the pipeline should include one test entity (such as Biomarker, Imaging_Test, Pathology_Test or Oncogene) and one result entity (such as Biomarker_Result, Pathology_Result or Tumor_Finding).

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

re_model = RelationExtractionModel.pretrained("re_oncology_test_result_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"]) \
    .setOutputCol("relation_extraction") \
    .setRelationPairs(["Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene", "Pathology_Test-Pathology_Result", "Pathology_Result-Pathology_Test"]) \
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

data = spark.createDataFrame([["Pathology showed tumor cells, which were positive for estrogen and progesterone receptors."]]).toDF("text")

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
    
val re_model = RelationExtractionModel.pretrained("re_oncology_test_result_wip", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunk", "dependencies"))
    .setOutputCol("relation_extraction")
    .setRelationPairs(Array("Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene", "Pathology_Test-Pathology_Result", "Pathology_Result-Pathology_Test"))
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

val data = Seq("Pathology showed tumor cells, which were positive for estrogen and progesterone receptors.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
   chunk1            entity1                   chunk2            entity2        relation   confidence
Pathology     Pathology_Test              tumor cells   Pathology_Result   is_finding_of   0.53310496
 positive   Biomarker_Result                 estrogen          Biomarker   is_finding_of   0.9453165
 positive   Biomarker_Result   progesterone receptors          Biomarker   is_finding_of   0.8816877
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_oncology_test_result_wip|
|Type:|re|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|266.9 KB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
        label  recall  precision   f1
            O    0.84       0.88 0.86
is_finding_of    0.89       0.85 0.87
    macro-avg    0.86       0.86 0.86
```
