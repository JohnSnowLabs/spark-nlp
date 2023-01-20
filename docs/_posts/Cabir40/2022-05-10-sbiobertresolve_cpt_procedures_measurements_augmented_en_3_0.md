---
layout: model
title: Sentence Entity Resolver for CPT codes (procedures and measurements) - Augmented
author: John Snow Labs
name: sbiobertresolve_cpt_procedures_measurements_augmented
date: 2022-05-10
tags: [licensed, en, clinical, entity_resolution, cpt]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.5.1
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model maps medical entities to CPT codes using Sentence Bert Embeddings. The corpus of this model has been extented to measurements, and this model is capable of mapping both procedures and measurement concepts/entities to CPT codes. Measurement codes are helpful in codifying medical entities related to tests and their results.


## Predicted Entities


`CPT Codes`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_CPT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_CPT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_measurements_augmented_en_3.5.1_3.0_1652168576968.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_measurements_augmented_en_3.5.1_3.0_1652168576968.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetectorDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("word_embeddings")

ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
.setInputCols(["sentence", "token", "word_embeddings"]) \
.setOutputCol("ner")\

ner_converter = NerConverterInternal()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")\
.setWhiteList(["Procedure", "Test"])

c2doc = Chunk2Doc()\
.setInputCols("ner_chunk")\
.setOutputCol("ner_chunk_doc") 

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sentence_embeddings")\
.setCaseSensitive(False)

cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_measurements_augmented", "en", "clinical/models")\
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("cpt_code")\
.setDistanceFunction("EUCLIDEAN")

resolver_pipeline = Pipeline(stages = [
document_assembler,
sentenceDetectorDL,
tokenizer,
word_embeddings,
ner,
ner_converter,
c2doc,
sbert_embedder,
cpt_resolver
])

model = resolver_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text='''She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. CT scan of the chest also revealed a large mediastinal lymph node. 
We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. 
At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy, which were performed, which revealed epithelioid malignant mesothelioma.'''

data = spark.createDataFrame([[text]]).toDF("text")

result = model.transform(data)
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

val ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")
.setInputCols(Array("sentence", "token", "word_embeddings"))
.setOutputCol("ner")

val ner_converter = new NerConverterInternal()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")
.setWhiteList(Array("Procedure", "Test"))

val c2doc = new Chunk2Doc()
.setInputCols(Array("ner_chunk"))
.setOutputCol("ner_chunk_doc") 

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sentence_embeddings")
.setCaseSensitive(False)

val cpt_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cpt_procedures_measurements_augmented", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sentence_embeddings"))
.setOutputCol("cpt_code")
.setDistanceFunction("EUCLIDEAN")

val resolver_pipeline = new PipelineModel().setStages(Array(
					    document_assembler, 
sentenceDetectorDL, 
tokenizer, 
word_embeddings, 
	                                    ner, 
ner_converter,  
c2doc, 
sbert_embedder, 
cpt_resolver))


val data = Seq("She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. CT scan of the chest also revealed a large mediastinal lymph node. We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy, which were performed, which revealed epithelioid malignant mesothelioma.").toDS.toDF("text")

val results = resolver_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.cpt.procedures_measurements").predict("""She was admitted to the hospital with chest pain and found to have bilateral pleural effusion, the right greater than the left. CT scan of the chest also revealed a large mediastinal lymph node. 
We reviewed the pathology obtained from the pericardectomy in March 2006, which was diagnostic of mesothelioma. 
At this time, chest tube placement for drainage of the fluid occurred and thoracoscopy, which were performed, which revealed epithelioid malignant mesothelioma.""")
```

</div>


## Results


```bash
+---------------------+---------+--------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
|                chunk|   entity|cpt_code|                                                                                   all_k_resolutions|                                                                                         all_k_codes|
+---------------------+---------+--------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+
| CT scan of the chest|     Test|   71250|Diagnostic CT scan of chest [Computed tomography, thorax, diagnostic; without contrast material]:...|71250:::70490:::76497:::71260:::74150:::70486:::73200:::70480:::77014:::73700:::71270:::70491:::7...|
|       pericardectomy|Procedure|   33030|Pericardectomy [Pericardiectomy, subtotal or complete; without cardiopulmonary bypass]:::Pericard...|33030:::33020:::64746:::49250:::27350:::68520:::32310:::27340:::33025:::32215:::41821:::1005708::...|
| chest tube placement|Procedure|   39503|Insertion of chest tube [Repair, neonatal diaphragmatic hernia, with or without chest tube insert...|39503:::96440:::32553:::35820:::32100:::36226:::21899:::29200:::0174T:::31502:::31605:::69424:::1...|
|drainage of the fluid|Procedure|   10140|Drainage of blood or fluid accumulation [Incision and drainage of hematoma, seroma or fluid colle...|10140:::40800:::61108:::41006:::62180:::83986:::49082:::27030:::21502:::49323:::32554:::51040:::6...|
|         thoracoscopy|Procedure| 1020900|Thoracoscopy [Thoracoscopy]:::Thoracoscopy, surgical; with control of traumatic hemorrhage | [Hea...|                   1020900:::32654:::32668:::1006014:::35820:::32606:::32555:::31781:::31515:::29200|
+---------------------+---------+--------+----------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------+


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_cpt_procedures_measurements_augmented|
|Compatibility:|Healthcare NLP 3.5.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[cpt_code]|
|Language:|en|
|Size:|360.4 MB|
|Case sensitive:|false|


## References


Trained on Current Procedural Terminology dataset with `sbiobert_base_cased_mli` sentence embeddings.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzkwMDEwNTIwLDE1Nzc1NjAxMzBdfQ==
-->
