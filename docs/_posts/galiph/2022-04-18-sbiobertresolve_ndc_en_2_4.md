---
layout: model
title: Sentence Entity Resolver for NDC (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_ndc
date: 2022-04-18
tags: [ndc, entity_resolution, licensed, en, clinical]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.2
spark_version: 2.4
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model maps clinical entities and concepts (like drugs/ingredients) to [National Drug Codes](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It also returns package options and alternative drugs in the all_k_aux_label column.


## Predicted Entities






{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_NDC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_NDC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_ndc_en_3.3.2_2.4_1650298194939.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")


sentenceDetector = SentenceDetectorDLModel.pretrained()\
	.setInputCols(["document"])\
	.setOutputCol("sentence")


tokenizer = Tokenizer()\
	.setInputCols(["sentence"])\
	.setOutputCol("token")


word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
	.setInputCols(["sentence", "token"])\
	.setOutputCol("embeddings")


posology_ner = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")


ner_converter = NerConverter() \
	.setInputCols(["sentence", "token", "ner"]) \
	.setOutputCol("ner_chunk")\
	.setWhiteList(["DRUG"])


c2doc = Chunk2Doc()\
	.setInputCols("ner_chunk")\
	.setOutputCol("ner_chunk_doc") 


sbert_embedder = BertSentenceEmbeddings\
	.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
	.setInputCols(["ner_chunk_doc"])\
	.setOutputCol("sentence_embeddings")


ndc_resolver = SentenceEntityResolverModel\
	.pretrained("sbiobertresolve_ndc", "en", "clinical/models") \
	.setInputCols(["ner_chunk", "sentence_embeddings"]) \
	.setOutputCol("ndc")\
	.setDistanceFunction("EUCLIDEAN")\
	.setCaseSensitive(False)


resolver_pipeline = Pipeline(stages = [
documentAssembler,
sentenceDetector,
tokenizer,
word_embeddings,
posology_ner,
ner_converter,
c2doc,
sbert_embedder,
ndc_resolver
])


data = spark.createDataFrame([["""The patient was given aspirin 81 mg and metformin 500 mg"""]]).toDF("text")

result = resolver_pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
	.setInputCol("text")
	.setOutputCol("document")


val sentenceDetector = SentenceDetectorDLModel.pretrained()
	.setInputCols("document")
	.setOutputCol("sentence")


val tokenizer = new Tokenizer()
	.setInputCols(Array("sentence"))
	.setOutputCol("token")


val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
	.setInputCols(Array("sentence", "token"))
	.setOutputCol("embeddings")


val clinical_ner = MedicalNerModel.pretrained("ner_posology_greedy", "en", "clinical/models")
	.setInputCols(Array("sentence", "token", "embeddings"))
	.setOutputCol("ner")


val ner_converter = new NerConverter()
	.setInputCols(Array("sentence", "token", "ner"))
	.setOutputCol("ner_chunk")
	.setWhiteList(Array("DRUG"))


val c2doc = new Chunk2Doc()
.setInputCols("ner_chunk")
.setOutputCol("ner_chunk_doc") 


val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk_doc"))
.setOutputCol("sentence_embeddings")


val ndc_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_ndc", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("ndc")
.setDistanceFunction("EUCLIDEAN")
.setCaseSensitive(False)


val resolver_pipeline = new Pipeline().setStages(Array(
documentAssembler,
sentenceDetector,
tokenizer,
word_embeddings,
posology_ner,
ner_converter,
c2doc,
sbert_embedder,
ndc_resolver
))

val clinical_note = Seq("The patient was given aspirin 81 mg and metformin 500 mg").toDS.toDF("text")

val results = resolver_pipeline.fit(clinical_note).transform(clinical_note)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.ndc").predict("""The patient was given aspirin 81 mg and metformin 500 mg""")
```

</div>


## Results


```bash
+----------------+----------+----------------------------------------------------------------------------------------------------+
|       ner_chunk|  ndc_code|                                                                                          aux_labels|
+----------------+----------+----------------------------------------------------------------------------------------------------+
|   aspirin 81 mg|41250-0780|{'packages': "['1 BOTTLE, PLASTIC in 1 PACKAGE (41250-780-01)  > 120 TABLET, DELAYED RELEASE in 1...|
|metformin 500 mg|62207-0491|{'packages': "['5000 TABLET in 1 POUCH (62207-491-31)', '25000 TABLET in 1 CARTON (62207-491-35)'...|
+----------------+----------+----------------------------------------------------------------------------------------------------+


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_ndc|
|Compatibility:|Healthcare NLP 3.3.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[ndc]|
|Language:|en|
|Size:|932.2 MB|
|Case sensitive:|false|


## References


It is trained on U.S. FDA 2022-NDC Codes dataset.
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNDQ1NzY1NzEsNzk2MjEwMTk5LC0xMj
UyOTg1MTkzLC0zMDMyNDAwMDEsNzk1OTY2MzYyLC03OTAwOTgx
NzBdfQ==
-->