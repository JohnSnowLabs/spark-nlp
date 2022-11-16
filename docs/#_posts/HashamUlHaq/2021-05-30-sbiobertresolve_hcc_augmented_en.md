---
layout: model
title: Sentence Entity Resolver for HCC codes (Augmented)
author: John Snow Labs
name: sbiobertresolve_hcc_augmented
date: 2021-05-30
tags: [entity_resolution, en, licensed]
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

This model maps extracted medical entities to HCC codes using Sentence Bert Embeddings.

## Predicted Entities

HCC codes and their descriptions.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_hcc_augmented_en_3.0.4_3.0_1622370690651.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use


```sbiobertresolve_hcc_augmented``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_clinical``` as NER model. ```PROBLEM``` set in ```.setWhiteList()```.


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
chunk2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
.setInputCols(["ner_chunk_doc"])\
.setOutputCol("sbert_embeddings")

resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_hcc_augmented","en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("resolution")\
.setDistanceFunction("EUCLIDEAN")

nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, resolver])

data = spark.createDataFrame([["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."]]).toDF("text")

results = nlpPipeline.fit(data).transform(data)
```
```scala
...
val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli","en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sbert_embeddings")

val resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_hcc_augmented","en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("resolution")
.setDistanceFunction("EUCLIDEAN")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter, chunk2doc, sbert_embedder, icdo_resolver))

val data = Seq.empty["This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."].toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.hcc").predict("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""")
```

</div>

## Results

```bash
+--------------------+-----+---+-------+----+----------+--------------------+--------------------+
|               chunk|begin|end| entity|code|confidence|   all_k_resolutions|         all_k_codes|
+--------------------+-----+---+-------+----+----------+--------------------+--------------------+
|        hypertension|   68| 79|PROBLEM| 139|    0.4357|renal hypertensio...|139:::85:::108:::...|
|chronic renal ins...|   83|109|PROBLEM| 139|    0.9748|chronic renal ins...|139:::140:::136::...|
|                COPD|  113|116|PROBLEM| 111|    0.5609|copd - chronic ob...| 111:::112:::84:::85|
|           gastritis|  120|128|PROBLEM| 188|    0.1991|functional disord...|188:::6:::75/18::...|
|                 TIA|  136|138|PROBLEM| 167|    0.3094|cerebral concussi...|167:::100:::167/1...|
|a non-ST elevatio...|  182|202|PROBLEM|  86|    0.4165|silent myocardial...|86:::87:::100:::9...|
|Guaiac positive s...|  208|229|PROBLEM| 188|    0.1492|appendicovesicost...|188:::33:::48:::1...|
|      mid LAD lesion|  332|345|PROBLEM|  86|    0.8090|stemi involving l...|      86:::108:::107|
|         hypotension|  362|372|PROBLEM|  59|    0.8107|drug-induced hypo...|59:::78:::2:::23:...|
|         bradycardia|  378|388|PROBLEM|  96|    0.5205|tachycardia-brady...|96:::59:::78:::23...|
|      vagal reaction|  466|479|PROBLEM| 108|    0.4985|vasomotor reactio...|108:::96:::23:::7...|
+--------------------+-----+---+-------+----+----------+--------------------+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_hcc_augmented|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[hcc_code]|
|Language:|en|
|Case sensitive:|false|