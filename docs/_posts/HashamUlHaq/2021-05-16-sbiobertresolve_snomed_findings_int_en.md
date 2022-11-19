---
layout: model
title: Sentence Entity Resolver for Snomed Concepts, INT version (``sbiobert_base_cased_mli`` embeddings)
author: John Snow Labs
name: sbiobertresolve_snomed_findings_int
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

This model maps extracted medical entities to Snomed codes (INT version) using using `sbiobert_base_cased_mli` Sentence Bert Embeddings, and has faster load time, with a speedup of about 6X when compared to previous versions. Also the load process now is more memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

## Predicted Entities

Predicts Snomed Codes and their normalized definition for each chunk.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_int_en_3.0.4_3.0_1621189624936.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

snomed_int_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_int","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("resolution")\
    .setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_int_resolver])

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

val snomed_int_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_int","en", "clinical/models")
    .setInputCols(Array("ner_chunk", "sbert_embeddings"))
    .setOutputCol("resolution")
    .setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, snomed_int_resolver))

val data = Seq("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.snomed.findings_int").predict("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""")
```

</div>

## Results

```bash
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
|               chunk|begin|end|   entity|           code|confidence|         resolutions|               codes|
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
|        hypertension|   68| 79|  PROBLEM|      266285003|    0.8867|rheumatic myocard...|266285003:::15529...|
|chronic renal ins...|   83|109|  PROBLEM|      236425005|    0.2470|chronic renal imp...|236425005:::90688...|
|                COPD|  113|116|  PROBLEM|      413839001|    0.0720|chronic lung dise...|413839001:::41384...|
|           gastritis|  120|128|  PROBLEM|      266502003|    0.3240|acute peptic ulce...|266502003:::45560...|
|                 TIA|  136|138|  PROBLEM|353101000119105|    0.0727|prostatic intraep...|353101000119105::...|
|a non-ST elevatio...|  182|202|  PROBLEM|      233843008|    0.2846|silent myocardial...|233843008:::71942...|
|Guaiac positive s...|  208|229|  PROBLEM|      168319009|    0.1167|stool culture pos...|168319009:::70396...|
|cardiac catheteri...|  295|317|     TEST|      301095005|    0.2137|cardiac finding::...|301095005:::25090...|
|                PTCA|  324|327|TREATMENT|842741000000109|    0.0631|occlusion of post...|842741000000109::...|
|      mid LAD lesion|  332|345|  PROBLEM|      449567000|    0.0808|overriding left v...|449567000:::25342...|
+--------------------+-----+---+---------+---------------+----------+--------------------+--------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_findings_int|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[snomed_int_code]|
|Language:|en|
|Case sensitive:|false|

## Data Source

Trained on SNOMED (INT version) Findings with ``sbiobert_base_cased_mli`` sentence embeddings.
http://www.snomed.org/