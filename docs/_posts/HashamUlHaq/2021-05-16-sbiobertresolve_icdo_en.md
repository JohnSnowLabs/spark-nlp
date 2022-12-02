---
layout: model
title: Sentence Entity Resolver for ICD-O (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_icdo
date: 2021-05-16
tags: [entity_resolution, clinical, licensed, en]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.0.4
spark_version: 3.0
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD-O codes (Topography & Morphology codes) using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

Given an oncological entity found in the text (via NER models like ner_jsl), it returns top terms and resolutions along with the corresponding ICD-O codes to present more granularity with respect to body parts mentioned. It also returns the original `Topography` codes, `Morphology` codes comprising of `Histology` and `Behavior` codes, and descriptions in the aux metadata.

## Predicted Entities

Predicts ICD-O Codes and their normalized definition for each chunk.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_icdo_en_3.0.4_3.0_1621191532225.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentence")

tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_clinical_large", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = NerConverter() \
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("ner_chunk")

chunk2doc = Chunk2Doc()\
    .setInputCols("ner_chunk")\
    .setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sbert_embeddings")

icdo_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icdo","en", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("resolution")\
    .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icdo_resolver])

data = spark.createDataFrame([["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
	.setInputCols("document")
	.setOutputCol("sentence")

val tokenizer = new Tokenizer()
	.setInputCols("sentence")
	.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_clinical_large", "en", "clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val chunk2doc = new Chunk2Doc()
    .setInputCols("ner_chunk")
    .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")
    .setInputCols("ner_chunk_doc")
    .setOutputCol("sbert_embeddings")

val icdo_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icdo","en", "clinical/models")
    .setInputCols(Array("ner_chunk", "sbert_embeddings"))
    .setOutputCol("resolution")
    .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icdo_resolver))

val data = Seq("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.icdo").predict("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""")
```

</div>

## Results

```bash
+--------------------+-----+---+---------+------+----------+--------------------+--------------------+
|               chunk|begin|end|   entity|  code|confidence|         resolutions|               codes|
+--------------------+-----+---+---------+------+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM|8312/3|    0.3558|Renal cell carcin...|8312/3:::9964/3::...|
|chronic renal ins...|   83|109|  PROBLEM|9980/3|    0.5290|Refractory anemia...|9980/3:::8312/3::...|
|                COPD|  113|116|  PROBLEM|9950/3|    0.2092|Polycythemia vera...|9950/3:::8141/3::...|
|           gastritis|  120|128|  PROBLEM|8150/3|    0.1795|Islet cell carcin...|8150/3:::8153/3::...|
|                 TIA|  136|138|  PROBLEM|9570/0|    0.4843|Neuroma, NOS:::Ca...|9570/0:::8692/3::...|
|a non-ST elevatio...|  182|202|  PROBLEM|8343/2|    0.1914|Non-invasive EFVP...|8343/2:::9150/0::...|
|Guaiac positive s...|  208|229|  PROBLEM|8155/3|    0.1069|Vipoma:::Myeloid ...|8155/3:::9930/3::...|
|cardiac catheteri...|  295|317|     TEST|8045/3|    0.1144|Combined small ce...|8045/3:::9705/3::...|
|                PTCA|  324|327|TREATMENT|9735/3|    0.0924|Plasmablastic lym...|9735/3:::9365/3::...|
|      mid LAD lesion|  332|345|  PROBLEM|9383/1|    0.0845|Subependymoma:::D...|9383/1:::8806/3::...|
+--------------------+-----+---+---------+------+----------+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_icdo|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[icdo_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on ICD-O Histology Behaviour dataset with ``sbiobert_base_cased_mli`` sentence embeddings.
https://apps.who.int/iris/bitstream/handle/10665/96612/9789241548496_eng.pdf