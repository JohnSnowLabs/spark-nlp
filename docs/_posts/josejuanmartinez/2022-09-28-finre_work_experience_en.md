---
layout: model
title: Financial Relation Extraction (Work Experience, Sm, Bidirectional)
author: John Snow Labs
name: finre_work_experience
date: 2022-09-28
tags: [work, experience, en, licensed]
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

This model allows you to analyzed present and past job positions of people, extracting relations between PERSON, ORG, ROLE and DATE. This model requires an NER with the mentioned entities, as `finner_org_per_role` and can also be combined with `finassertiondl_past_roles` to detect if the entities are mentioned to have happened in the PAST or not (although you can also infer that from the relations as `had_role_until`).

This model is a `sm` model without meaningful directions in the relations (the model was not trained to understand if the direction of the relation is from left to right or right to left). There are bigger models in Models Hub trained also with directed relationships.

## Predicted Entities

`has_role`, `had_role_until`, `has_role_from`, `works_for`, `has_role_in_company`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINRE_WORK_EXPERIENCE){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finre_work_experience_en_1.0.0_3.0_1664360618647.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finre_work_experience_en_1.0.0_3.0_1664360618647.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")\

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.NerModel.pretrained('finner_org_per_role_date', 'en', 'finance/models')\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence","token","ner"])\
    .setOutputCol("ner_chunk")

pos = nlp.PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos")

dependency_parser = nlp.DependencyParserModel().pretrained("dependency_conllu", "en")\
    .setInputCols(["sentence", "pos", "token"])\
    .setOutputCol("dependencies")

re_ner_chunk_filter = finance.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunk")\
    .setRelationPairs(["PERSON-ROLE, ORG-ROLE, DATE-ROLE, PERSON-ORG"])\
    .setMaxSyntacticDistance(5)

re_Model = finance.RelationExtractionDLModel.pretrained("finre_work_experience", "en", "finance/models")\
    .setInputCols(["re_ner_chunk", "sentence"])\
    .setOutputCol("relations")\
    .setPredictionThreshold(0.5)

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter,
    pos,
    dependency_parser,
    re_ner_chunk_filter,
    re_Model
])

empty_df = spark.createDataFrame([['']]).toDF("text")

re_model = pipeline.fit(empty_df)

light_model = LightPipeline(re_model)

text_list = ["""We have experienced significant changes in our senior management team over the past several years, including the appointments of Mark Schmitz as our Executive Vice President and Chief Operating Officer in 2019.""",
             """In January 2019, Jose Cil was assigned the CEO of Restaurant Brands International, and Daniel Schwartz was assigned the Executive Chairman of the company.""",
             ]

results = light_model.fullAnnotate(text_list)
```

</div>

## Results

```bash
has_role	    PERSON	129	140	Mark Schmitz	ROLE	149	172	Executive Vice President	0.8707728
has_role	    PERSON	129	140	Mark Schmitz	ROLE	178	200	Chief Operating Officer	0.97559035
has_role_from	ROLE	149	172	Executive Vice President	DATE	205	208	2019	0.9327241
has_role_from	ROLE	178	200	Chief Operating Officer	DATE	205	208	2019	0.90718126
has_role_from	DATE	3	14	January 2019	ROLE	43	45	CEO	0.996639
has_role_from	DATE	3	14	January 2019	ROLE	120	137	Executive Chairman	0.9964874
has_role	    PERSON	17	24	Jose Cil	ROLE	43	45	CEO	0.8917691
has_role	    PERSON	17	24	Jose Cil	ROLE	120	137	Executive Chairman	0.8527716
has_role	    ROLE	43	45	CEO	PERSON	87	101	Daniel Schwartz	0.5765097
has_role	    PERSON	87	101	Daniel Schwartz	ROLE	120	137	Executive Chairman	0.79235893
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finre_work_experience|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|409.9 MB|

## References

Manual annotations on CUAD dataset, 10K filings and Wikidata

## Benchmarking

```bash
 label                Recall   Precision  F1       Support
 had_role_until       0.972    0.972      0.972    36  
 has_role             0.986    0.980      0.983    146 
 has_role_from        0.983    0.983      0.983    58  
 has_role_in_company  0.954    0.969      0.961    65  
 works_for            0.933    0.933      0.933    15  
 Avg.                 0.966    0.967      0.966     -   
 Weighted-Avg.        0.975    0.975      0.975     -   
```  
