---
layout: model
title: Sentence Entity Resolver for RxNorm (disposition)
author: John Snow Labs
name: sbiobertresolve_rxnorm_disposition
date: 2021-08-12
tags: [rxnorm, licensed, en, entity_resolution]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.1.3
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps medication entities (like drugs/ingredients) to RxNorm codes and their dispositions using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

## Predicted Entities

Predicts RxNorm Codes, their normalized definition for each chunk, and dispositions if any. In the result, look for the aux_label parameter in the metadata to get dispositions divided by `|`.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_RXNORM/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/ER_RXNORM.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_disposition_en_3.1.3_2.4_1628792971821.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_disposition_en_3.1.3_2.4_1628792971821.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

```sbiobertresolve_rxnorm_disposition``` resolver model must be used with ```sbiobert_base_cased_mli``` as embeddings ```ner_posology``` as NER model. ```DRUG``` set in ```.setWhiteList()```.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sbert_embeddings")

rxnorm_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm_disposition", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sbert_embeddings"]) \
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

pipelineModel = PipelineModel(
stages = [
documentAssembler,
sbert_embedder,
rxnorm_resolver
])

rxnorm_lp = LightPipeline(pipelineModel)
result = rxnorm_lp.fullAnnotate("belimumab 80 mg/ml injectable solution")
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("sbert_embeddings")

val rxnorm_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm_disposition", "en", "clinical/models")
.setInputCols(Array("ner_chunk", "sbert_embeddings"))
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val pipelineModel= new PipelineModel().setStages(Array(documentAssembler, sbert_embedder, rxnorm_resolver))

val rxnorm_lp = LightPipeline(pipelineModel)
val result = rxnorm_lp.fullAnnotate("belimumab 80 mg/ml injectable solution")
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm_disposition").predict("""belimumab 80 mg/ml injectable solution""")
```

</div>

## Results

```bash
|    | chunks                                | code    | resolutions                                                                                                                                                                                 | all_codes                                         | all_k_aux_labels                                                                            | all_distances                                 |
|---:|:--------------------------------------|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|:--------------------------------------------------------------------------------------------|:----------------------------------------------|
|  0 |belimumab 80 mg/ml injectable solution | 1092440 | [belimumab 80 mg/ml injectable solution, belimumab 80 mg/ml injectable solution [benlysta], ifosfamide 80 mg/ml injectable solution, belimumab 80 mg/ml [benlysta], belimumab 80 mg/ml, ...]| [1092440, 1092444, 107034, 1092442, 1092438, ...] | [Immunomodulator, Immunomodulator, Alkylating agent, Immunomodulator, Immunomodulator, ...] | [0.0000, 0.0145, 0.0479, 0.0619, 0.0636, ...] |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_disposition|
|Compatibility:|Healthcare NLP 3.1.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Case sensitive:|false|
