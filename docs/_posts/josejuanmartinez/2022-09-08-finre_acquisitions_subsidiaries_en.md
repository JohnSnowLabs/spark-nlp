---
layout: model
title: Acquisitions / Subsidiaries Relation Extraction
author: John Snow Labs
name: finre_acquisitions_subsidiaries
date: 2022-09-08
tags: [en, finance, re, relations, acquisitions, subsidiaries, licensed]
task: Relation Extraction
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a WIP model, what means it's on early stage and will be improved as more data comes in from in-house annotations.

The aim of this model is to retrieve acquisition or subsidiary relationships between Organizations, included when the acquisition was carried out ("was_acquired") and by whom ("was_acquired_by"). Subsidiaries are tagged with the relationship "is_subsidiary_of".

## Predicted Entities

`was_acquired`, `was_acquired_by`, `is_subsidiary_of`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINRE_ACQUISITIONS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_acquisitions_subsidiaries_en_1.0.0_3.2_1662641362605.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
        .setInputCols(["document"])\
        .setOutputCol("sentence")
        
tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

# ==========
# This is needed only to filter relation pairs using RENerChunksFilter (see below)
# ==========
pos = PerceptronModel.pretrained("pos_anc", 'en')\
          .setInputCols("sentence", "token")\
          .setOutputCol("pos")

depency_parser = DependencyParserModel.pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos", "token"]) \
    .setOutputCol("dependencies")
# ==========

# ==========
# USE ANY NERMODEL WHICH RETRIEVES ORG
  We recommend `finner_orgs_prods_alias` because it retrieves also companies Aliases (as "AWS" in the sentence "Amazon Web Services (AWS)")
# ==========
bert_embeddings= BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("bert_embeddings")

ner_model_org= FinanceNerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
        .setInputCols(["sentence", "token", "bert_embeddings"])\
        .setOutputCol("ner_cuad")

ner_converter_org = NerConverter()\
        .setInputCols(["sentence","token","ner_cuad"])\
        .setOutputCol("ner_chunk_org")\
        .setWhiteList(['ORG', 'PRODUCT', 'ALIAS'])

# ==========
# USE ANY NERMODEL WHICH RETRIEVES DATES. 
  In this example, we will go for big accuracy with large Roberta Ontonotes mode
# ==========
roberta_embeddings = RoBertaEmbeddings.pretrained('roberta_large', 'en')\
      .setInputCols(["token", "sentence"])\
      .setOutputCol("roberta_embeddings")

ner_model_onto = NerDLModel.pretrained('ner_ontonotes_roberta_large', 'en') \
    .setInputCols(['sentence', 'token', 'roberta_embeddings']) \
    .setOutputCol('ner_onto')

# ==========

ner_converter_onto = NerConverter()\
        .setInputCols(["sentence","token","ner_onto"])\
        .setOutputCol("ner_chunk_onto")\
        .setWhiteList(["DATE"])

chunk_merger = ChunkMergeApproach()\
        .setInputCols('ner_chunk_org', "ner_chunk_onto")\
        .setOutputCol('ner_chunk')

re_ner_chunk_filter = RENerChunksFilter() \
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(["DATE-ORG", "DATE-ALIAS", "DATE-PRODUCT", "ORG-ORG"])

re_Model = RelationExtractionDLModel.pretrained("finre_acquisitions_subsidiaries", "en", "finance/models")\
        .setInputCols(["re_ner_chunk", "sentence"])\
        .setOutputCol("relations")\
        .setPredictionThreshold(0.5)

pipeline = Pipeline(stages=[
        documentAssembler,
        sentence_detector,
        tokenizer,
        bert_embeddings,
        pos,
        ner_model_org,
        depency_parser,
        ner_converter_org,
        roberta_embeddings,
        ner_model_onto,
        ner_converter_onto,
        chunk_merger,
        re_ner_chunk_filter,
        re_Model
        ])
empty_df = spark.createDataFrame([['']]).toDF("text")

re_model = pipeline.fit(empty_df)

light_model = LightPipeline(re_model)

light_model.fullAnnotate("""On January 15, 2020, Cadence acquired all of the outstanding equity of AWR Corporation ("AWR"). On February 6, 2020, Cadence also acquired all of the outstanding equity of Integrand Software, Inc. ("Integrand").""")
```

</div>

## Results

```bash
relation	entity1	entity1_begin	entity1_end	chunk1	entity2	entity2_begin	entity2_end	chunk2	confidence
0	was_acquired	DATE	3	18	January 15, 2020	ORG	21	27	Cadence	0.9996277
1	was_acquired	DATE	3	18	January 15, 2020	ORG	71	85	AWR Corporation	0.9994985
2	was_acquired	DATE	3	18	January 15, 2020	ALIAS	89	91	AWR	0.99967
3	was_acquired_by	ORG	21	27	Cadence	ORG	71	85	AWR Corporation	0.7940858
4	was_acquired	DATE	99	114	February 6, 2020	ORG	117	123	Cadence	0.999689
5	was_acquired	DATE	99	114	February 6, 2020	ORG	172	195	Integrand Software, Inc.	0.99955875
6	was_acquired	DATE	99	114	February 6, 2020	ALIAS	199	207	Integrand	0.99918574
7	was_acquired_by	ORG	117	123	Cadence	ORG	172	195	Integrand Software, Inc.	0.6435062
...
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_acquisitions_subsidiaries|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.9 MB|

## References

Manual annotations on CUAD dataset, 10K filings and Wikidata

## Benchmarking

```bash
Relation           Recall Precision        F1   Support

is_subsidiary_of     0.836     0.924     0.878       146
no_rel              0.968     0.932     0.950       684
was_acquired        0.936     0.944     0.940       218
was_acquired_by     0.857     0.911     0.883       168

Avg.                0.899     0.928     0.913

Weighted Avg.       0.931     0.931     0.930
```
