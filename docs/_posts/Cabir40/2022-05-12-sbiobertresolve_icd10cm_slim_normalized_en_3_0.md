---
layout: model
title: ICD10CM Sentence Entity Resolver (Slim, normalized)
author: John Snow Labs
name: sbiobertresolve_icd10cm_slim_normalized
date: 2022-05-12
tags: [licensed, clinical, en, entity_resolution, icd10]
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


This model maps clinical entities and concepts to ICD10 CM codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. In this model, synonyms having low cosine similarity to unnormalized terms are dropped, making the model slim. It also returns the official resolution text within the brackets inside the metadata


## Predicted Entities


`ICD10 CM Codes`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_CM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_ICD10_CM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icd10cm_slim_normalized_en_3.5.1_3.0_1652337920061.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


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
    
icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_normalized", "en", "clinical/models") \
    .setInputCols(["ner_chunk", "sentence_embeddings"]) \
    .setOutputCol("icd_code")\
    .setDistanceFunction("EUCLIDEAN")\
    .setReturnCosineDistances(True)
    
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
    
val resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_normalized", "en", "clinical/models")
      .setInputCols(Array("ner_chunk", "sentence_embeddings"))
      .setOutputCol("icd_code")
      .setDistanceFunction("EUCLIDEAN")
      .setReturnCosineDistances(True)

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
</div>


## Results


```bash
+-------------------------------------+-------+--------+---------------------------------------------------------------------------------+---------------------------------------------------+
|                                chunk| entity|icd_code|                                                                all_k_resolutions|                                        all_k_codes|
+-------------------------------------+-------+--------+---------------------------------------------------------------------------------+---------------------------------------------------+
|        gestational diabetes mellitus|PROBLEM|   E11.9|gestational diabetes mellitus [Type 2 diabetes mellitus without complications]...|E11.9:::O24.41:::O24.919:::O24.419:::O24.439:::....|
|subsequent type two diabetes mellitus|PROBLEM|  O24.11|pre-existing type 2 diabetes mellitus [Pre-existing type 2 diabetes mellitus, ...|O24.11:::E11.8:::E11.9:::E11:::E13.9:::E11.3:::....|
|                                 T2DM|PROBLEM|   E11.9|t2dm [Type 2 diabetes mellitus without complications]:::gm>2 [GM2 gangliosidos...|E11.9:::E75.00:::H35.89:::F80.0:::R44.8:::M79.89...|
|             HTG-induced pancreatitis|PROBLEM| F10.988|alcohol-induced pancreatitis [Alcohol use, unspecified with other alcohol-indu...|F10.988:::K85.9:::K85.3:::K85:::K85.2:::K85.8:::...|
|                      acute hepatitis|PROBLEM|   B17.9|acute hepatitis [Acute viral hepatitis, unspecified]:::acute hepatitis [Acute ...|B17.9:::K72.0:::B15.9:::B15:::B17.2:::Z03.89:::....|
|                              obesity|PROBLEM|   E66.8|abdominal obesity [Other obesity]:::overweight and obesity [Overweight and obe...|E66.8:::E66:::E66.01:::E66.9:::Z91.89:::E66.3:::...|
|                             polyuria|PROBLEM|     R35|polyuria [Polyuria]:::nocturnal polyuria [Nocturnal polyuria]:::other polyuria...|R35:::R35.81:::R35.89:::R31:::R30.0:::E72.01:::....|
|                           polydipsia|PROBLEM|   R63.1|polydipsia [Polydipsia]:::psychogenic polydipsia [Other impulse disorders]:::p...|R63.1:::F63.89:::O40.9XX0:::O40:::G47.50:::G47.5...|
|                        poor appetite|PROBLEM|   R63.0|poor appetite [Anorexia]:::patient dissatisfied with nutrition regime [Persons...|R63.0:::Z76.89:::R53.1:::R10.9:::R45.81:::R44.8:...|
|                             vomiting|PROBLEM|   R11.1|vomiting [Vomiting]:::vomiting [Vomiting, unspecified]:::intermittent vomiting...|R11.1:::R11.10:::R11:::G43.A:::G43.A0:::R11.0:::...|
|        a respiratory tract infection|PROBLEM|   J06.9|upper respiratory tract infection [Acute upper respiratory infection, unspecif...|J06.9:::T17:::T17.9:::J04.10:::J22:::J98.8:::J98.9.|
+-------------------------------------+-------+--------+---------------------------------------------------------------------------------+---------------------------------------------------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icd10cm_slim_normalized|
|Compatibility:|Healthcare NLP 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10cm_code]|
|Language:|en|
|Size:|846.3 MB|
|Case sensitive:|false|
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDIzMjU0MjAzLC0xNjMwNjI1OTcxXX0=
-->