---
layout: model
title: Acquisitions / Subsidiaries Relation Extraction (md, Unidirectional)
author: John Snow Labs
name: finre_acquisitions_subsidiaries_md
date: 2022-11-08
tags: [acquisition, subsidiaries, en, licensed]
task: Relation Extraction
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RelationExtractionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

IMPORTANT: Don't run this model on the whole financial report. Instead:
- Split by paragraphs;
- Use the `finclf_acquisitions_item` Text Classifier to select only these paragraphs;
 
This model is a `md` model, meaning that the directions in the relations are meaningful: `chunk1` is the source of the relation, `chunk2` is the target.

The aim of this model is to retrieve acquisition or subsidiary relationships between Organizations, included when the acquisition was carried out ("was_acquired") and by whom ("was_acquired_by"). Subsidiaries are tagged with the relationship "is_subsidiary_of".

## Predicted Entities

`was_acquired`, `was_acquired_by`, `is_subsidiary_of`, `other`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINRE_ACQUISITIONS/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_acquisitions_subsidiaries_md_en_1.0.0_3.0_1667920790547.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentencizer = nlp.SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"])\
        .setOutputCol("sentence")
                      
tokenizer = nlp.Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

bert_embeddings= nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("bert_embeddings")

ner_model_date = finance.NerModel.pretrained("finner_sec_dates", "en", "finance/models")\
        .setInputCols(["sentence", "token", "bert_embeddings"])\
        .setOutputCol("ner_dates")

ner_converter_date = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner_dates"])\
        .setOutputCol("ner_chunk_date")

ner_model_org= finance.NerModel.pretrained("finner_orgs_prods_alias", "en", "finance/models")\
        .setInputCols(["sentence", "token", "bert_embeddings"])\
        .setOutputCol("ner_orgs")

ner_converter_org = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner_orgs"])\
        .setOutputCol("ner_chunk_org")\
        .setWhiteList(['ORG', 'PRODUCT', 'ALIAS'])

chunk_merger = finance.ChunkMergeApproach()\
        .setInputCols('ner_chunk_org', "ner_chunk_date")\
        .setOutputCol('ner_chunk')

reDL = finance.RelationExtractionDLModel().pretrained('finre_acquisitions_subsidiaries_md', 'en', 'finance/models')\
    .setInputCols(["ner_chunk", "sentence"])\
    .setOutputCol("relations")

nlpPipeline = nlp.Pipeline(stages=[
        documentAssembler,
        sentencizer,
        tokenizer,
        bert_embeddings,
        ner_model_date,
        ner_converter_date,
        ner_model_org,
        ner_converter_org,
        chunk_merger,
        reDL])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = "Whatsapp, Inc. was acquired by Meta, Inc"

lmodel = LightPipeline(model)
results = lmodel.fullAnnotate(text)
rel_df = get_relations_df (results)
rel_df = rel_df[rel_df['relation']!='no_rel']
print(rel_df.to_string(index=False))
```

</div>

## Results

```bash
        relation entity1 entity1_begin entity1_end          chunk1 entity2 entity2_begin entity2_end chunk2 confidence
 was_acquired_by     ORG             0          13  Whatsapp, Inc.     ORG            31          34   Meta  0.9527305
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_acquisitions_subsidiaries_md|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.7 MB|

## References

In-house annotations on SEC 10K filings and Wikidata

## Benchmarking

```bash
label                         Recall Precision  F1       Support
is_subsidiary_of     0.583     0.618     0.600        36
other                        0.975     0.948     0.961       243
was_acquired         0.836     0.895     0.864        61
was_acquired_by   0.767     0.780     0.773        60
Avg.                          0.790     0.810     0.800        406
Weighted-Avg.        0.887     0.885     0.886        406
```