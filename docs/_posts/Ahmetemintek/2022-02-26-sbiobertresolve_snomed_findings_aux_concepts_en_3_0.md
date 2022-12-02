---
layout: model
title: Sentence Entity Resolver for SNOMED Concepts
author: John Snow Labs
name: sbiobertresolve_snomed_findings_aux_concepts
date: 2022-02-26
tags: [snomed, licensed, en, clinical, aux, ct]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.2
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps clinical entities and concepts to Snomed codes using sbiobert_base_cased_mli Sentence Bert Embeddings. This is also capable of extracting `Morph Abnormality`, `Procedure`, `Substance`, `Physical Object`, and `Body Structure` concepts of Snomed codes.

In the metadata, the `all_k_aux_labels` can be divided to get further information: `ground truth`, `concept`, and `aux`. For example, in the example shared below the ground truth is `Atherosclerosis`, concept is `Observation`, and aux is `Morph Abnormality`

## Predicted Entities

`SNOMED Codes`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_snomed_findings_aux_concepts_en_3.1.2_3.0_1645879611162.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenizer = Tokenizer() \
      .setInputCols(["sentence"]) \
      .setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en", "clinical/models")\
      .setInputCols(["sentence", "token"])\
      .setOutputCol("embeddings")

ner_clinical = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("clinical_ner")

ner_converter = NerConverter() \
      .setInputCols(["sentence", "token", "clinical_ner"]) \
      .setOutputCol("ner_chunk")\

chunk2doc = Chunk2Doc() \
      .setInputCols("ner_chunk") \
      .setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")\
     .setDistanceFunction("EUCLIDEAN")

nlpPipeline= Pipeline(stages=[
                              documentAssembler,
                              sentenceDetector,
                              tokenizer,
                              word_embeddings,
                              ner_clinical,
                              ner_converter,
                              chunk2doc,
                              sbert_embedder,
                              snomed_resolver
])

text= """FINDINGS: The patient was found upon excision of the cyst that it contained a large Prolene suture; beneath this was a very small incisional hernia, the hernia cavity, which contained omentum; the hernia was easily repaired"""

df= spark.createDataFrame([[text]]).toDF("text")

result= nlpPipeline.fit(df).transform(df)
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

val tokenizer = Tokenizer() 
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en", "clinical/models")
      .setInputCols(Array("sentence", "token"))
      .setOutputCol("embeddings")

val ner_clinical = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models") 
      .setInputCols(Array("sentence", "token", "embeddings")) 
      .setOutputCol("clinical_ner")

val ner_converter = NerConverter() 
      .setInputCols(Array("sentence", "token", "clinical_ner")) 
      .setOutputCol("ner_chunk")

val chunk2doc = Chunk2Doc() 
      .setInputCols("ner_chunk") 
      .setOutputCol("ner_chunk_doc")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models")
      .setInputCols("ner_chunk_doc")
      .setOutputCol("sbert_embeddings")

val snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models")
     .setInputCols(Array("ner_chunk", "sbert_embeddings"))
     .setOutputCol("snomed_code")

val new nlpPipeine().setStages(Array(documentAssembler,
                                    sentenceDetector,
                                    tokenizer,
                                    word_embeddings,
                                    ner_clinical,
                                    ner_converter,
                                    chunk2doc,
                                    sbert_embedder,
                                    snomed_resolver))

val text= """FINDINGS: The patient was found upon excision of the cyst that it contained a large Prolene suture; beneath this was a very small incisional hernia, the hernia cavity, which contained omentum; the hernia was easily repaired"""

val df = Seq(text).toDF(“text”) 

val result= nlpPipeline.fit(df).transform(df)
```
</div>

## Results

```bash
| sent_id 	| ner_chunk                      	| entity    	| snomed_code       	| all_codes                                                              	| resolutions                                                                                                                                                                                       	| all_k_aux_labels                                                                                                                                                                                                                                                       	|
|---------	|--------------------------------	|-----------	|-------------------	|------------------------------------------------------------------------	|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| 1       	| excision                       	| TREATMENT 	| 180397004         	| [180397004, 65801008, 129304002, 257819000, 82868003,...               	| [excision from organ noc: [sinus tract] or [fistula], excision, excision - action, surgical excision, margins of excision,...                                                                     	| [Excision from organ NOC: [sinus tract] or [fistula]\|Procedure\|Procedure, Excision\|Procedure\|Procedure, Excision - action\|Observation\|Qualifier Value, Surgical excision\|Observation\|Qualifier Value, Surgical margins\|Spec Anatomic Site\|Body Structure,... 	|
| 1       	| the cyst                       	| PROBLEM   	| 246178003         	| [246178003, 103552005, 441457006, 264515009, 367643001,...             	| [form of cyst, cyst, cyst, cyst, cyst,...                                                                                                                                                         	| [Form of cyst\|Observation\|Attribute, Kingdom Protozoa cyst\|Observation\|Organism, Cyst\|Condition\|Clinical Finding, Cyst - morphology\|Observation\|Qualifier Value, Cyst\|Observation\|Morph Abnormality,...                                                      	|
| 1       	| a large Prolene suture         	| TREATMENT 	| 20594411000001105 	| [20594411000001105, 7267511000001100, 20125511000001105, 463182000,... 	| [finger stalls plastic medium, portia disposable gloves polythene medium (bray group ltd), silk mittens 8-14 years, polybutester suture, skinnies silk gloves large child blue (dermacea ltd),... 	| [-\|-\|-, Portia disposable gloves polythene medium (Bray Group Ltd)\|Device\|Physical Object, -\|-\|-, Polybutester suture\|Device\|Physical Object, -\|-\|-,...                                                                                                      	|
| 1       	| a very small incisional hernia 	| PROBLEM   	| 155752004         	| [155752004, 196894007, 266513000, 415772007, 266514006,                	| [simple incisional hernia, simple incisional hernia, simple incisional hernia, uncomplicated ventral incisional hernia, umbilical hernia - simple,...                                             	| [Hernia - incisional\|Condition\|Clinical Finding, Uncomplicated incisional hernia\|Condition\|Clinical Finding, Incisional hernia - simple\|Condition\|Clinical Finding, Uncomplicated ventral incisional hernia\|Condition\|Clinical Finding,...                     	|
| 1       	| the hernia cavity              	| PROBLEM   	| 112639008         	| [112639008, 52515009, 359801000, 414403008, 147780008,                 	| [protrusion, hernia, hernia, hernia, notification of whooping cough,...                                                                                                                           	| [Protrusion\|Observation\|Morph Abnormality, Hernia of abdominal cavity\|Condition\|Clinical Finding, Hernia of abdominal cavity\|Condition\|Clinical Finding, Hernia\|Observation\|Morph Abnormality,...                                                              	|
| 1       	| the hernia                     	| PROBLEM   	| 52515009          	| [52515009, 359801000, 414403008, 147780008, 112639008,                 	| [hernia, hernia, hernia, notification of whooping cough, protrusion,...                                                                                                                           	| [Hernia of abdominal cavity\|Condition\|Clinical Finding, Hernia of abdominal cavity\|Condition\|Clinical Finding, Hernia\|Observation\|Morph Abnormality, Notification of whooping cough\|Procedure\|Procedure,...                                                    	|
| 1       	| repaired                       	| TREATMENT 	| 50826004          	| [50826004, 4365001, 257903006, 33714007, 260938008,                    	| [repaired, repair, repair, corrected, restoration,...                                                                                                                                             	| [Repaired\|Observation\|Qualifier Value, Surgical repair\|Procedure\|Procedure, Repair - action\|Observation\|Qualifier Value, Corrected\|Observation\|Qualifier Value, Type of restoration\|Observation\|Attribute,...                                                	|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_snomed_findings_aux_concepts|
|Compatibility:|Healthcare NLP 3.1.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk, sbert_embeddings]|
|Output Labels:|[snomed_code]|
|Language:|en|
|Size:|4.7 GB|
|Case sensitive:|false|
