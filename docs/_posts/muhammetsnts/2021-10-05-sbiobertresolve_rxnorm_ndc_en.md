---
layout: model
title: Sentence Entity Resolver for RxNorm (NDC)
author: John Snow Labs
name: sbiobertresolve_rxnorm_ndc
date: 2021-10-05
tags: [licensed, clinical, en, ndc, rxnorm]
task: Entity Resolution
language: en
edition: Healthcare NLP 3.2.3
spark_version: 2.4
supported: true
annotator: SentenceEntityResolverModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model maps `DRUG ` entities to RxNorm codes and their [National Drug Codes (NDC)](https://www.drugs.com/ndc.html#:~:text=The%20NDC%2C%20or%20National%20Drug,and%20the%20commercial%20package%20size.) using `sbiobert_base_cased_mli ` sentence embeddings. You can find all NDC codes of drugs seperated by `|` symbol in the all_k_aux_labels parameter of the metadata.

## Predicted Entities

`RxNorm Codes`, `NDC Codes`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/ER_RXNORM_NDC/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_ndc_en_3.2.3_2.4_1633424811842.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/sbiobertresolve_rxnorm_ndc_en_3.2.3_2.4_1633424811842.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("ner_chunk")

sbert_embedder = BertSentenceEmbeddings\
.pretrained('sbiobert_base_cased_mli', 'en','clinical/models')\
.setInputCols(["ner_chunk"])\
.setOutputCol("sentence_embeddings")

rxnorm_ndc_resolver = SentenceEntityResolverModel\
.pretrained("sbiobertresolve_rxnorm_ndc", "en", "clinical/models") \
.setInputCols(["ner_chunk", "sentence_embeddings"]) \
.setOutputCol("rxnorm_code")\
.setDistanceFunction("EUCLIDEAN")

rxnorm_ndc_pipeline = Pipeline(
stages = [
documentAssembler,
sbert_embedder,
rxnorm_ndc_resolver])

data = spark.createDataFrame([["activated charcoal 30000 mg powder for oral suspension"]]).toDF("text")

res = rxnorm_ndc_pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
.setInputCol("text")
.setOutputCol("ner_chunk")

val sbert_embedder = BertSentenceEmbeddings
.pretrained("sbiobert_base_cased_mli", "en","clinical/models")
.setInputCols(Array("ner_chunk")
.setOutputCol("sentence_embeddings")

val rxnorm_ndc_resolver = SentenceEntityResolverModel
.pretrained("sbiobertresolve_rxnorm_ndc", "en", "clinical/models") 
.setInputCols(Array("ner_chunk", "sentence_embeddings")) 
.setOutputCol("rxnorm_code")
.setDistanceFunction("EUCLIDEAN")

val rxnorm_ndc_pipeline = new Pipeline().setStages(Array(documentAssembler, sbert_embedder, rxnorm_ndc_resolver))

val data = Seq("activated charcoal 30000 mg powder for oral suspension").toDF("text")

val res = rxnorm_ndc_pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.resolve.rxnorm_ndc").predict("""activated charcoal 30000 mg powder for oral suspension""")
```

</div>

## Results

```bash
+--+------------------------------------------------------+-----------+-----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+
|  |ner_chunk                                             |rxnorm_code|all_codes                                      |resolutions                                                                                                                                                                                                                                                                               |all_k_aux_labels (ndc_codes)                                                                           |
+--+------------------------------------------------------+-----------+-----------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------+
|0 |activated charcoal 30000 mg powder for oral suspension|1440919    |[1440919, 808917, 1088194, 1191772, 808921,...]|'activated charcoal 30000 MG Powder for Oral Suspension', 'Activated Charcoal 30000 MG Powder for Oral Suspension', 'wheat dextrin 3000 MG Powder for Oral Solution [Benefiber]', 'cellulose 3000 MG Oral Powder [Unifiber]', 'fosfomycin 3000 MG Powder for Oral Solution [Monurol]', ...|69784030828, 00395052791, 08679001362|86790016280|00067004490, 46017004408|68220004416, 00456430001,...|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sbiobertresolve_rxnorm_ndc|
|Compatibility:|Healthcare NLP 3.2.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[rxnorm_code]|
|Language:|en|
|Case sensitive:|false|
