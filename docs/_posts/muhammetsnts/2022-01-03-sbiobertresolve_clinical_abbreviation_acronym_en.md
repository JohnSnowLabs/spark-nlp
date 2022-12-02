---
layout: model
title: Sentence Entity Resolver for Clinical Abbreviations and Acronyms (sbiobert_base_cased_mli embeddings)
author: John Snow Labs
name: sbiobertresolve_clinical_abbreviation_acronym
date: 2022-01-03
tags: [abbreviation, entity_resolver, licensed, en, clinical, acronym]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.3.4
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical abbreviations and acronyms to their meanings using `sbiobert_base_cased_mli` Sentence Bert Embeddings. It is the first primitive version of abbreviation resolution and will be improved further in the following releases.

## Predicted Entities

`Abbreviation Meanings`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_clinical_abbreviation_acronym_en_3.3.4_2.4_1641239524364.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

tokenizer = Tokenizer()\
.setInputCols(["document"])\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["document", "token"])\
.setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") \
.setInputCols(["document", "token", "word_embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverterInternal() \
.setInputCols(["document", "token", "ner"]) \
.setOutputCol("ner_chunk")\
.setWhiteList(['ABBR'])

sentence_chunk_embeddings = BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")\
.setInputCols(["document", "ner_chunk"])\
.setOutputCol("sentence_embeddings")\
.setChunkWeight(0.5)\
.setCaseSensitive(True)


abbr_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("abbr_meaning")\
.setDistanceFunction("EUCLIDEAN")\


resolver_pipeline = Pipeline(
stages = [
document_assembler,
tokenizer,
word_embeddings,
clinical_ner,
ner_converter,
sentence_chunk_embeddings,
abbr_resolver
])

text = "The patient admitted from the IR for aggressive irrigation of the Miami pouch. DISCHARGE DIAGNOSES: 1. A 58-year-old female with a history of stage 2 squamous cell carcinoma of the cervix status post total pelvic exenteration in 1991."

sample_text = spark.createDataFrame([[text]]).toDF('text')

abbr_result = resolver_pipeline.fit(sample_text).transform(sample_text)

```
```scala
val document_assembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val tokenizer = Tokenizer()
.setInputCols(Array("document"))
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("document", "token"))
.setOutputCol("word_embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_abbreviation_clinical", "en", "clinical/models") 
.setInputCols(Array("document", "token", "word_embeddings")) 
.setOutputCol("ner")

val ner_converter = NerConverterInternal() 
.setInputCols(Array("document", "token", "ner")) 
.setOutputCol("ner_chunk")
.setWhiteList(Array("ABBR"))

val sentence_chunk_embeddings = BertSentenceChunkEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models")
.setInputCols(Array("document", "ner_chunk"))
.setOutputCol("sentence_embeddings")
.setChunkWeight(0.5)
.setCaseSensitive(True)


val abbr_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_clinical_abbreviation_acronym", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("abbr_meaning")
.setDistanceFunction("EUCLIDEAN")


val resolver_pipeline = new Pipeline().setStages(document_assembler, tokenizer, word_embeddings, clinical_ner, ner_converter, sentence_chunk_embeddings, abbr_resolver)

val sample_text = Seq("The patient admitted from the IR for aggressive irrigation of the Miami pouch. DISCHARGE DIAGNOSES: 1. A 58-year-old female with a history of stage 2 squamous cell carcinoma of the cervix status post total pelvic exenteration in 1991.").toDF("text")

val abbr_result = resolver_pipeline.fit(sample_text).transform(sample_text)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.clinical_abbreviation_acronym").predict("""The patient admitted from the IR for aggressive irrigation of the Miami pouch. DISCHARGE DIAGNOSES: 1. A 58-year-old female with a history of stage 2 squamous cell carcinoma of the cervix status post total pelvic exenteration in 1991.""")
```

</div>

## Results

```bash
+-------+---------+------+------------------------+-------------------------------------------------------------------------+-----------------+---------------------------------+
|sent_id|ner_chunk|entity|            abbr_meaning|                                                            all_k_results|all_k_resolutions|           all_k_cosine_distances|
+-------+---------+------+------------------------+-------------------------------------------------------------------------+-----------------+---------------------------------+
|      0|       IR|  ABBR|interventional radiology|interventional radiology:::immediate-release:::(stage) IA:::intraarterial|IR:::IR:::IA:::IA|0.0156:::0.0945:::0.1046:::0.1111|
+-------+---------+------+------------------------+-------------------------------------------------------------------------+-----------------+---------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_clinical_abbreviation_acronym|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[abbr_meaning]|
|Language:|en|
|Size:|105.3 MB|
|Case sensitive:|false|
