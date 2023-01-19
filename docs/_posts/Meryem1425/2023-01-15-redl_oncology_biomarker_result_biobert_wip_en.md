---
layout: model
title: Relation Extraction between Biomarkers and Results (ReDL)
author: John Snow Labs
name: redl_oncology_biomarker_result_biobert_wip
date: 2023-01-15
tags: [licensed, clinical, oncology, en, relation_extraction, test, biomarker, tensorflow]
task: Relation Extraction
language: en
edition: Healthcare NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This relation extraction model links Biomarker and Oncogene extractions to their corresponding Biomarker_Result extractions.

## Predicted Entities

`is_finding_of`, `O`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_ONCOLOGY/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/27.Oncology_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_oncology_biomarker_result_biobert_wip_en_4.2.4_3.0_1673766618517.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/redl_oncology_biomarker_result_biobert_wip_en_4.2.4_3.0_1673766618517.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_converter = NerConverterInternal() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")
          
pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

re_ner_chunk_filter = RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setMaxSyntacticDistance(10)\
    .setRelationPairs(["Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene"])

re_model = RelationExtractionDLModel.pretrained("redl_oncology_biomarker_result_biobert_wip", "en", "clinical/models")\
    .setInputCols(["re_ner_chunk", "sentence"])\
    .setOutputCol("relation_extraction")
        
pipeline = Pipeline(stages=[document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter,
                            pos_tagger,
                            dependency_parser,
                            re_ner_chunk_filter,
                            re_model])

data = spark.createDataFrame([["Immunohistochemistry was negative for thyroid transcription factor-1 and napsin A. The test was positive for ER and PR, and negative for HER2."]]).toDF("text")

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
    
val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pos_tagger = PerceptronModel.pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("pos_tags")
    
val dependency_parser = DependencyParserModel.pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentence", "pos_tags", "token"))
    .setOutputCol("dependencies")

val re_ner_chunk_filter = new RENerChunksFilter()
    .setInputCols(Array("ner_chunk", "dependencies"))
    .setOutputCol("re_ner_chunk")
    .setMaxSyntacticDistance(10)
    .setRelationPairs(Array("Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene"))

val re_model = RelationExtractionDLModel.pretrained("redl_oncology_biomarker_result_biobert_wip", "en", "clinical/models")
    .setInputCols(Array("re_ner_chunk", "sentence"))
    .setOutputCol("relation_extraction")

val pipeline = new Pipeline().setStages(Array(document_assembler,
                            sentence_detector,
                            tokenizer,
                            word_embeddings,
                            ner,
                            ner_converter,
                            pos_tagger,
                            dependency_parser,
                            re_ner_chunk_filter,
                            re_model))

val data = Seq("Immunohistochemistry was negative for thyroid transcription factor-1 and napsin A. The test was positive for ER and PR, and negative for HER2.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+-------------+----------------+-------------+-----------+--------+----------------+-------------+-----------+--------------------+----------+
|     relation|         entity1|entity1_begin|entity1_end|  chunk1|         entity2|entity2_begin|entity2_end|              chunk2|confidence|
+-------------+----------------+-------------+-----------+--------+----------------+-------------+-----------+--------------------+----------+
|is_finding_of|Biomarker_Result|           25|         32|negative|       Biomarker|           38|         67|thyroid transcrip...|0.99808085|
|is_finding_of|Biomarker_Result|           25|         32|negative|       Biomarker|           73|         78|              napsin|0.99637383|
|is_finding_of|Biomarker_Result|           96|        103|positive|       Biomarker|          109|        110|                  ER|0.99221414|
|is_finding_of|Biomarker_Result|           96|        103|positive|       Biomarker|          116|        117|                  PR| 0.9893672|
|            O|Biomarker_Result|           96|        103|positive|        Oncogene|          137|        140|                HER2| 0.9986272|
|            O|       Biomarker|          109|        110|      ER|Biomarker_Result|          124|        131|            negative| 0.9999089|
|            O|       Biomarker|          116|        117|      PR|Biomarker_Result|          124|        131|            negative| 0.9998932|
|is_finding_of|Biomarker_Result|          124|        131|negative|        Oncogene|          137|        140|                HER2|0.98810333|
+-------------+----------------+-------------+-----------+--------+----------------+-------------+-----------+--------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|redl_oncology_biomarker_result_biobert_wip|
|Compatibility:|Healthcare NLP 4.2.4+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|401.7 MB|

## References

In-house annotated oncology case reports.

## Benchmarking

```bash
        label  recall  precision   f1  
            O    0.93       0.97 0.95   
is_finding_of    0.97       0.93 0.95    
    macro-avg    0.95       0.95 0.95
```
