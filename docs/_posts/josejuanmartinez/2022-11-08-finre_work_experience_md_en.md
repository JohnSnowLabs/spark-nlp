---
layout: model
title: Financial Relation Extraction (Work Experience, Medium, Unidirectional)
author: John Snow Labs
name: finre_work_experience_md
date: 2022-11-08
tags: [work, experience, role, en, licensed]
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
- Split by paragraphs. You can use [notebook 1](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL) in Finance or Legal as inspiration;
- Use the `finclf_work_experience_item` Text Classifier to select only these paragraphs; 

This is a `md` (medium) version of `finre_work_experience` model, trained with more data and with **unidirectional relation extractions**, meaning now the direction of the arrow matters: it goes from the source (`chunk1`) to the target (`chunk2`).

This model allows you to analyzed present and past job positions of people, extracting relations between PERSON, ORG, ROLE and DATE. This model requires an NER with the mentioned entities, as `finner_org_per_role_date` and can also be combined with `finassertiondl_past_roles` to detect if the entities are mentioned to have happened in the PAST or not (although you can also infer that from the relations as `had_role_until`).

## Predicted Entities

`has_role`, `had_role_until`, `has_role_from`, `works_for`, `has_role_in_company`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINRE_WORK_EXPERIENCE){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_work_experience_md_en_1.0.0_3.0_1667922980930.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

sentencizer = SentenceDetectorDLModel\
        .pretrained("sentence_detector_dl", "en") \
        .setInputCols(["document"])\
        .setOutputCol("sentence")                     
                     
tokenizer = Tokenizer()\
        .setInputCols(["sentence"])\
        .setOutputCol("token")

bert_embeddings= BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en")\
        .setInputCols(["sentence", "token"])\
        .setOutputCol("bert_embeddings")

ner_model = finance.NerModel.pretrained("finner_org_per_role_date", "en", "finance/models")\
        .setInputCols(["sentence", "token", "bert_embeddings"])\
        .setOutputCol("ner_orgs")

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner_orgs"])\
        .setOutputCol("ner_chunk")

pos = PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos")

dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos", "token"])\
    .setOutputCol("dependencies")

re_filter = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(["PERSON-ROLE", "PERSON-ORG", "ORG-ROLE", "DATE-ROLE"])
                            
reDL = finance.RelationExtractionDLModel()\
    .pretrained('finre_work_experience_md','en','finance/models')\
    .setInputCols(["re_ner_chunk", "sentence"])\
    .setOutputCol("relations")

nlpPipeline = Pipeline(stages=[
        documentAssembler,
        sentencizer,
        tokenizer,
        bert_embeddings,
        ner_model,
        ner_converter,
        pos,
        dependency_parser,
        re_filter,
        reDL])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text = f"On December 15, 2021, Anirudh Devgan assumed the role of President and Chief Executive Officer of Cadence, replacing Lip-Bu Tan. Prior to his role as Chief Executive Officer, Dr. Devgan served as President of Cadence. Concurrently, Mr. Tan transitioned to the role of Executive Chair"

lmodel = LightPipeline(model)
results = lmodel.fullAnnotate(text)
rel_df = get_relations_df (results)
rel_df = rel_df[rel_df['relation']!='other']
print(rel_df.to_string(index=False))
print()
```

</div>

## Results

```bash
           relation entity1 entity1_begin entity1_end                  chunk1 entity2 entity2_begin entity2_end                  chunk2 confidence
      has_role_from    DATE             3          19       December 15, 2021    ROLE            57          65               President  0.9532135
      has_role_from    DATE             3          19       December 15, 2021    ROLE            71          93 Chief Executive Officer 0.91833746
           has_role  PERSON            22          35          Anirudh Devgan    ROLE            57          65               President  0.9993814
           has_role  PERSON            22          35          Anirudh Devgan    ROLE            71          93 Chief Executive Officer  0.9889985
          works_for  PERSON            22          35          Anirudh Devgan     ORG            98         104                 Cadence  0.9983778
has_role_in_company    ROLE            57          65               President     ORG            98         104                 Cadence  0.9997348
has_role_in_company    ROLE            71          93 Chief Executive Officer     ORG            98         104                 Cadence 0.99845624
           has_role    ROLE           150         172 Chief Executive Officer  PERSON           175         184              Dr. Devgan 0.85268635
has_role_in_company    ROLE           150         172 Chief Executive Officer     ORG           209         215                 Cadence  0.9976404
           has_role  PERSON           175         184              Dr. Devgan    ROLE           196         204               President 0.99899226
          works_for  PERSON           175         184              Dr. Devgan     ORG           209         215                 Cadence 0.99876934
has_role_in_company    ROLE           196         204               President     ORG           209         215                 Cadence  0.9997203
           has_role  PERSON           232         238                 Mr. Tan    ROLE           268         282         Executive Chair 0.98612714
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
label             Recall Precision        F1   Support
had_role_until      1.000     1.000     1.000       117
has_role            0.998     0.995     0.997       649
has_role_from       1.000     1.000     1.000       401
has_role_in_company     0.993     0.993     0.993       268
other               0.996     0.996     0.996       235
works_for           0.994     1.000     0.997       330
Avg.                0.997     0.997     0.997    2035
Weighted-Avg.       0.997     0.997     0.997   2035
```
