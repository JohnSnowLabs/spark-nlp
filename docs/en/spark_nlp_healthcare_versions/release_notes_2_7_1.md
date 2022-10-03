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


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_7_0">Version 2.7.0</a>
    </li>
    <li>
        <strong>Version 2.7.1</strong>
    </li>
    <li>
        <a href="release_notes_2_7_2">Version 2.7.2</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li class="active"><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>