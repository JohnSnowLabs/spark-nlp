---
layout: model
title: Sentence Entity Resolver for Billable ICD10-CM HCC Codes (sbiobertresolve_icd10cm_slim_billable_hcc)
author: John Snow Labs
name: sbiobertresolve_icd10cm_slim_billable_hcc
date: 2022-05-11
tags: [licensed, en, clinical, icd10, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.5.1
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model maps extracted clinical entities to ICD-10-CM codes using `sbiobert_base_cased_mli` sentence bert embeddings. In this model, synonyms having low cosine similarity to unnormalized terms are dropped. It returns the official resolution text within the brackets and also provides billable and HCC information of the codes in `all_k_aux_labels` parameter in the metadata. This column can be divided to get further details: `billable status || hcc status || hcc score`. For example, if `all_k_aux_labels` is like `1||1||19` which means the `billable status` is 1, `hcc status` is 1, and `hcc score` is 19.


## Predicted Entities


`ICD10 Codes`, `billable status`, `hcc status`, `hcc score`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_slim_billable_hcc_en_3.5.1_3.0_1652294908790.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_slim_billable_hcc_en_3.5.1_3.0_1652294908790.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")


sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
.setInputCols(["document"])\
.setOutputCol("sentence")


tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")


word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("word_embeddings")


ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token", "word_embeddings"])\
.setOutputCol("ner")\


ner_converter = NerConverterInternal()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")\
.setWhiteList(["PROBLEM"])


c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 


sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")\
.setCaseSensitive(False)

icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("icd_code")\
.setDistanceFunction("EUCLIDEAN")

resolver_pipeline = Pipeline(
stages = [
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
ner,
ner_converter,
c2doc,
sbert_embedder,
icd_resolver
])

data = spark.createDataFrame([["""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis and obesity , presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection."""]]).toDF("text")

result = resolver_pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")


val sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
.setInputCols(Array("document"))
.setOutputCol("sentence")


val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("word_embeddings")


val ner = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "word_embeddings"))
.setOutputCol("ner")


val ner_converter = new NerConverterInternal()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")
.setWhiteList(Array("PROBLEM"))


val c2doc = new Chunk2Doc()
.setInputCols(Array("ner_chunk"))
.setOutputCol("ner_chunk_doc") 


val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sentence_embeddings")
.setCaseSensitive(False)

val resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sentence_embeddings"))
.setOutputCol("icd_code")
.setDistanceFunction("EUCLIDEAN")



val resolver_pipeline = new PipelineModel().setStages(Array(document_assembler, 
sentenceDetectorDL, 
tokenizer, 
word_embeddings, 
ner, 
ner_converter,  
c2doc, 
sbert_embedder, 
resolver))


val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis and obesity , presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.").toDS.toDF("text")


val results = resolver_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icd10cm.slim_billable_hcc").predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus (T2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with acute hepatitis and obesity , presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting. Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.""")
```

</div>


## Results


```bash
+-------------------------------------+-------+--------+-------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------+
|                                chunk| entity|icd_code|                                                  all_k_resolutions|                                       all_k_codes|                                      all_k_aux_labels |
+-------------------------------------+-------+--------+-------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------+
|        gestational diabetes mellitus|PROBLEM|  O24.41|gestational diabetes mellitus [Gestational diabetes mellitus in ...|O24.41:::E11.9:::O24.919:::O24.419:::O24.439:::...| 0||0||0:::1||1||19:::1||0||0:::1||0||0:::1||0||0:::...|
|subsequent type two diabetes mellitus|PROBLEM|  O24.11|pre-existing type 2 diabetes mellitus [Pre-existing type 2 diabe...|O24.11:::E11.8:::E11.9:::E11:::E13.9:::E11.3:::...| 0||0||0:::1||1||18:::1||1||19:::0||0||0:::1||1||19:...|
|                                 T2DM|PROBLEM|   E11.9|t2dm [Type 2 diabetes mellitus without complications]:::gm>2 [GM...|E11.9:::E75.00:::H35.89:::F80.0:::R44.8:::M79.8...| 1||1||19:::1||1||52:::1||0||0:::1||0||0:::1||0||0::...|
|             HTG-induced pancreatitis|PROBLEM|   K85.9|alcohol-induced pancreatitis [Acute pancreatitis, unspecified]:::..|K85.9:::F10.988:::K85.3:::K85:::K85.2:::K85.8::...| 0||0||0:::1||1||55:::0||0||0:::0||0||0:::0||0||0:::...|
|                      acute hepatitis|PROBLEM|   K72.0|acute hepatitis [Acute and subacute hepatic failure]:::acute hep...|K72.0:::B17.9:::B15.9:::B15:::B17.2:::Z03.89:::...| 0||0||0:::1||0||0:::1||0||0:::0||0||0:::1||0||0:::1...|
|                              obesity|PROBLEM|   E66.8|abdominal obesity [Other obesity]:::overweight and obesity [Over...|E66.8:::E66:::E66.01:::E66.9:::Z91.89:::E66.3::...| 1||0||0:::0||0||0:::1||1||22:::1||0||0:::1||0||0:::...|
|                             polyuria|PROBLEM|     R35|polyuria [Polyuria]:::nocturnal polyuria [Nocturnal polyuria]:::...|R35:::R35.81:::R35.89:::R31:::R30.0:::E72.01:::...| 0||0||0:::1||0||0:::1||0||0:::0||0||0:::1||0||0:::1...|
|                           polydipsia|PROBLEM|   R63.1|polydipsia [Polydipsia]:::psychogenic polydipsia [Other impulse ...|R63.1:::F63.89:::O40:::O40.9XX0:::G47.50:::G47....| 1||0||0:::1||0||0:::0||0||0:::1||0||0:::1||0||0:::0...|
|                        poor appetite|PROBLEM|   R63.0|poor appetite [Anorexia]:::patient dissatisfied with nutrition r...|R63.0:::Z76.89:::R53.1:::R10.9:::R45.81:::R44.8...| 1||0||0:::1||0||0:::1||0||0:::1||0||0:::1||0||0:::1...|
|                             vomiting|PROBLEM|   R11.1|vomiting [Vomiting]:::vomiting [Vomiting, unspecified]:::intermi...|R11.1:::R11.10:::R11:::G43.A0:::G43.A:::R11.0::...| 0||0||0:::1||0||0:::0||0||0:::1||0||0:::0||0||0:::1...|
|        a respiratory tract infection|PROBLEM|   J06.9|upper respiratory tract infection [Acute upper respiratory infec...|J06.9:::T17.9:::T17:::J04.10:::J22:::J98.8:::J9...| 1||0||0:::0||0||0:::0||0||0:::1||0||0:::1||0||0:::1...|
+-------------------------------------+-------+--------+----------------------------------------------------------------------------------------------------+--------------------------------------------------+-------------------------------------------------------+


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10cm_slim_billable_hcc|
|Compatibility:|Healthcare NLP 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Size:|846.6 MB|
|Case sensitive:|false|
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1Nzk5NzQxNTcsLTE0MjY2MTg4OTNdfQ
==
-->