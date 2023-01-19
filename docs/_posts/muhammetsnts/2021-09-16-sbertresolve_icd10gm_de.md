---
layout: model
title: Sentence Entity Resolver for ICD10-GM
author: John Snow Labs
name: sbertresolve_icd10gm
date: 2021-09-16
tags: [icd10gm, en, clinical, licensed, de]
task: Entity Resolution
language: de
edition: Healthcare NLP 3.2.2
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps extracted medical entities to ICD10-GM codes for the German language using `sent_bert_base_cased` (de) embeddings.

## Predicted Entities

`ICD10GM Codes`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_ICD10_GM_DE/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbertresolve_icd10gm_de_3.2.2_2.4_1631814227170.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbertresolve_icd10gm_de_3.2.2_2.4_1631814227170.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

icd10gm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_icd10gm", "de", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("icd10gm_code")

icd10gm_pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
icd10gm_resolver])

icd_lp = LightPipeline(icd10gm_pipelineModel)
```
```scala
val documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings.pretrained("sent_bert_base_cased", "de")\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

val icd10gm_resolver = SentenceEntityResolverModel.pretrained("sbertresolve_icd10gm", "de", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("icd10gm_code")

val icd10gm_pipelineModel = new PipelineModel().setStages(Array(documentAssembler,sbert_embedder,icd10gm_resolver))

val icd_lp = LightPipeline(icd10gm_pipelineModel)
```


{:.nlu-block}
```python
import nlu
nlu.load("de.resolve.icd10gm").predict("""Put your text here.""")
```

</div>

## Results

```bash
|    | chunks  | code    | resolutions                                                                                                                                                                                                                                                                                                                                      | all_codes                                                                                                                                                       | all_distances                                                                                                                                                                                            |
|---:|:--------|:--------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | Dyspnoe | C671    |Dyspnoe, Schlafapnoe, Dysphonie, Frühsyphilis, Hyperzementose, Hypertrichose, Makrostomie, Dystonie, Nokardiose, Lebersklerose, Dyspareunie, Schizophrenie, Skoliose, Dysurie, Diphyllobothriose, Heterophorie, Rektozele, Enophthalmus, Amyloidose, Hyperventilation, Neurasthenie, Sarkoidose, Psoriasis-Arthropathie, Hyperodontie, Enteroptose| [R06.0, G47.3, R49.0, A51, K03.4, L68, Q18.4, G24, A43, K74.1, N94.1, F20, M41, R30.0, B70.0, H50.5, N81.6, H05.4, E85, R06.4, F48.0, D86, L40.5, K00.1, K63.4] | [0.0000, 2.5602, 3.0529, 3.3310, 3.4645, 3.7148, 3.7568, 3.8115, 3.8557, 3.8577, 3.9448, 3.9681, 3.9799, 3.9889, 4.0036, 4.0773, 4.0825, 4.1342, 4.2031, 4.2155, 4.2313, 4.2341, 4.2775, 4.2802, 4.2823] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbertresolve_icd10gm|
|Compatibility:|Healthcare NLP 3.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[icd10_gm2022_de_code]|
|Language:|de|
|Case sensitive:|false|