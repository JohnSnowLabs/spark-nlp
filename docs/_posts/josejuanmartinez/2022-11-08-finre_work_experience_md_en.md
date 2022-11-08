---
layout: model
title: Financial Relation Extraction (Work Experience, Md, Unidirectional)
author: John Snow Labs
name: finre_work_experience_md
date: 2022-11-08
tags: [work, experience, roles, en, licensed]
task: Relation Extraction
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model allows you to analyzed present and past job positions of people, extracting relations between `PERSON`, `ORG`, `ROLE` and `DATE`. This model requires an NER with the mentioned entities, as `finner_org_per_role_date` and can also be combined with `finassertiondl_past_roles` to detect if the entities are mentioned to have happened in the PAST or not (although you can also infer that from the relations as `had_role_until`).

This is a `md` model with Unidirectional Relations, meaning that the model retrieves in chunk1 the left side of the relation (source), and in chunk2 the right side (target).

## Predicted Entities

`has_role`, `had_role_until`, `has_role_from`, `works_for`, `has_role_in_company`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINRE_EXPERIENCES/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_work_experience_md_en_1.0.0_3.0_1667901262440.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
        .setInputCols(["sentence", "token", "bert_embeddings"])\
        .setOutputCol("ner_orgs")

ner_converter = nlp.NerConverter()\
        .setInputCols(["sentence","token","ner_orgs"])\
        .setOutputCol("ner_chunk_org")

chunk_merger = finance.ChunkMergeApproach()\
        .setInputCols('ner_chunk_org', "ner_chunk_date")\
        .setOutputCol('ner_chunk')

pos = nlp.PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos")

dependency_parser = nlp.DependencyParserModel().pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos", "token"])\
    .setOutputCol("dependencies")

re_filter = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(["PERSON-ROLE", "PERSON-ORG", "ORG-ROLE", "DATE-ROLE"])\
    .setMaxSyntacticDistance(10)

reDL = finance.RelationExtractionDLModel()\
    .pretrained("finre_work_experience_md","en", "finance/models")\
    .setInputCols(["re_ner_chunk", "sentence"])\
    .setOutputCol("relations")\
    .setPredictionThreshold(0.85)

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentencizer,
        tokenizer,
        bert_embeddings,
        ner_model_date,
        ner_converter_date,
        ner_model,
        ner_converter,
        chunk_merger,
        pos,
        dependency_parser,
        re_filter,
        reDL])

text = """We have experienced significant changes in our senior management team over the past several years, including the appointments of Mark Schmitz as our Executive Vice President and Chief Operating Officer in 2019."""

df = spark.createDataFrame([text]).toDF("text")

model = pipeline.fit(df)

result = model.transform(df)
```

</div>

## Results

```bash

| relation      | entity1 | entity1_begin | entity1_end | chunk1                   | entity2 | entity2_begin | entity2_end | chunk2                   | confidence |
|---------------|---------|---------------|-------------|--------------------------|---------|---------------|-------------|--------------------------|------------|
| has_role      | PERSON  | 129           | 140         | Mark Schmitz             | ROLE    | 149           | 172         | Executive Vice President | 0.9945273  |
| has_role      | PERSON  | 129           | 140         | Mark Schmitz             | ROLE    | 178           | 200         | Chief Operating Officer  | 0.9947194  |
| has_role_from | ROLE    | 149           | 172         | Executive Vice President | DATE    | 205           | 208         | 2019                     | 0.9985196  |
| has_role_from | ROLE    | 178           | 200         | Chief Operating Officer  | DATE    | 205           | 208         | 2019                     | 0.99905354 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_work_experience_md|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.7 MB|

## References

Manual annotations on CUAD dataset, 10K filings and Wikidata

## Benchmarking

```bash
label       Recall Precision        F1   Support
had_role_until      0.992     0.992     0.992       124
has_role            1.000     0.998     0.999       642
has_role_from       0.997     0.997     0.997       399
has_role_in_company     0.996     1.000     0.998       268
other               1.000     0.996     0.998       237
works_for           0.994     0.997     0.995       330
Avg.                0.997     0.997     0.997 2030
Weighted-Avg.       0.998     0.998     0.997 2030
```