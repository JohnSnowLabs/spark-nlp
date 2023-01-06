---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.7.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_7_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="prev_ver h3-box" markdown="1">

### 2.7.1

We are glad to announce that Spark NLP for Healthcare 2.7.1 has been released !

In this release, we introduce the following features:

#### 1. Sentence BioBert and Bluebert Transformers that are fine tuned on [MedNLI](https://physionet.org/content/mednli/) dataset.

Sentence Transformers offers a framework that provides an easy method to compute dense vector representations for sentences and paragraphs (also known as sentence embeddings). The models are based on BioBert and BlueBert, and are tuned specifically to meaningful sentence embeddings such that sentences with similar meanings are close in vector space. These are the first PyTorch based models we managed to port into Spark NLP.

Here is how you can load these:
```python
sbiobert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
```
```python
sbluebert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbluebert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
```

#### 2. SentenceEntityResolvers powered by s-Bert embeddings.

The advent of s-Bert sentence embeddings changed the landscape of Clinical Entity Resolvers completely in Spark NLP. Since s-Bert is already tuned on MedNLI (medical natural language inference) dataset, it is now capable of populating the chunk embeddings in a more precise way than before.

Using sbiobert_base_cased_mli, we trained the following Clinical Entity Resolvers:

sbiobertresolve_icd10cm  
sbiobertresolve_icd10pcs  
sbiobertresolve_snomed_findings (with clinical_findings concepts from CT version)  
sbiobertresolve_snomed_findings_int  (with clinical_findings concepts from INT version)  
sbiobertresolve_snomed_auxConcepts (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from CT version)  
sbiobertresolve_snomed_auxConcepts_int  (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from INT version)  
sbiobertresolve_rxnorm  
sbiobertresolve_icdo  
sbiobertresolve_cpt  

Code sample:

(after getting the chunk from ChunkConverter)

```python
c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

snomed_ct_resolver = SentenceEntityResolverModel
 .pretrained("sbiobertresolve_snomed_findings","en", "clinical/models") \
 .setInputCols(["ner_chunk", "sbert_embeddings"]) \
 .setOutputCol("snomed_ct_code")\
 .setDistanceFunction("EUCLIDEAN")
 ```

Output:

|    | chunks                      |   begin |   end |      code | resolutions
|  2 | COPD  				 |     113 |   116 |  13645005 | copd - chronic obstructive pulmonary disease
|  8 | PTCA                        |     324 |   327 | 373108000 | post percutaneous transluminal coronary angioplasty (finding)
| 16 | close monitoring            |     519 |   534 | 417014005 | on examination - vigilance


See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb#scrollTo=VtDWAlnDList) for details.

#### 3. We are releasing the following pretrained clinical NER models:

ner_drugs_large   
(trained with medications dataset, and extracts drugs with the dosage, strength, form and route at once as a single entity; entities: drug)  
ner_deid_sd_large  
(extracts PHI entities, trained with augmented dataset)  
ner_anatomy_coarse  
(trained with enriched anatomy NER dataset; entities: anatomy)  
ner_anatomy_coarse_biobert  
chunkresolve_ICD10GM_2021 (German ICD10GM resolver)


We are also releasing two new NER models:

ner_aspect_based_sentiment  
(extracts positive, negative and neutral aspects about restaurants from the written feedback given by reviewers. )  
ner_financial_contract  
(extract financial entities from contracts. See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/19.Financial_Contract_NER.ipynb) for details.)  


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}