---
layout: model
title: Sentence Entity Resolver for CPT codes (Augmented)
author: John Snow Labs
name: sbiobertresolve_cpt_procedures_augmented
date: 2021-05-30
tags: [licensed, entity_resolution, en, clinical]
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

This model maps extracted medical entities to CPT codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. This model is enriched with augmented data for better performance.

## Predicted Entities

CPT codes and their descriptions.

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_cpt_procedures_augmented_en_3.0.4_3.0_1622371775342.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



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

val resolver = SentenceEntityResolverModel\
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
nlu.load("en.resolve.cpt.procedures_augmented").predict("""This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .""")
```

</div>

## Results

```bash
+--------------------+-----+---+-------+-----+----------+--------------------+--------------------+
|               chunk|begin|end| entity| code|confidence|   all_k_resolutions|         all_k_codes|
+--------------------+-----+---+-------+-----+----------+--------------------+--------------------+
|        hypertension|   68| 79|PROBLEM|36440|    0.3349|Hypertransfusion:...|36440:::24935:::0...|
|chronic renal ins...|   83|109|PROBLEM|50395|    0.0821|Nephrostomy:::Ren...|50395:::50328:::5...|
|                COPD|  113|116|PROBLEM|32960|    0.1575|Lung collapse pro...|32960:::32215:::1...|
|           gastritis|  120|128|PROBLEM|43501|    0.1772|Gastric ulcer sut...|43501:::43631:::4...|
|                 TIA|  136|138|PROBLEM|61460|    0.1432|Intracranial tran...|61460:::64742:::2...|
|a non-ST elevatio...|  182|202|PROBLEM|61624|    0.1151|Percutaneous non-...|61624:::61626:::3...|
|Guaiac positive s...|  208|229|PROBLEM|44005|    0.1115|Enterolysis:::Abd...|44005:::49080:::4...|
|      mid LAD lesion|  332|345|PROBLEM|0281T|    0.2407|Plication of left...|0281T:::93462:::9...|
|         hypotension|  362|372|PROBLEM|99135|    0.9935|Induced hypotensi...|99135:::99185:::9...|
|         bradycardia|  378|388|PROBLEM|99135|    0.3884|Induced hypotensi...|99135:::33305:::3...|
|      vagal reaction|  466|479|PROBLEM|55450|    0.1427|Vasoligation:::Va...|55450:::64408:::7...|
+--------------------+-----+---+-------+-----+----------+--------------------+--------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_cpt_procedures_augmented|
|Compatibility:|Healthcare NLP 3.0.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[cpt_code_aug]|
|Language:|en|
|Case sensitive:|false|