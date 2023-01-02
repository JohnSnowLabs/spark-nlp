---
layout: model
title: Notice Clause Relation Extraction Model
author: John Snow Labs
name: legre_notice_clause_xs
date: 2022-12-17
tags: [en, legal, relations, redl, licensed, tensorflow]
task: Relation Extraction
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Relation Extraction model aimed to be used in notice clauses, to retrieve relations between entities as NOTICE_PARTY, ADDRESS, EMAIL, TITLE etc. Make sure you run this model only on the NER entities in notice clauses, after you filter them using `legclf_notice_clause`

## Predicted Entities

`has_notice_party`, `has_address`, `has_person`, `has_phone`, `has_fax`, `has_title`, `has_email`, `has_department`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legre_notice_clause_xs_en_1.0.0_3.0_1671280929569.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

sentence_detector = nlp.SentenceDetectorDLModel.pretrained()\
    .setInputCols("document")\
    .setOutputCol("sentence")\
    .setCustomBounds(["\n\n"])\
    .setUseCustomBoundsOnly(True)

tokenizer = nlp.Tokenizer()\
    .setInputCols("sentence")\
    .setOutputCol("token")

pos_tagger = nlp.PerceptronModel.pretrained()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("pos_tags")

dependency_parser = nlp.DependencyParserModel() \
    .pretrained("dependency_conllu", "en") \
    .setInputCols(["sentence", "pos_tags", "token"]) \
    .setOutputCol("dependencies")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = legal.NerModel.pretrained('legner_notice_clause', 'en', 'legal/models') \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

ner_converter = nlp.NerConverter() \
    .setInputCols(["sentence","token","ner"]) \
    .setOutputCol("ner_chunk")

re_filter = legal.RENerChunksFilter()\
    .setInputCols(["ner_chunk", "dependencies"])\
    .setOutputCol("re_ner_chunks")\
    .setMaxSyntacticDistance(12)\
    .setRelationPairs(['NAME-NOTICE_PARTY','NAME-ADDRESS','NAME-PERSON', 'NAME-TITLE','NAME-EMAIL','NAME-PHONE', 'NAME-FAX', 'NAME-DEPARTMENT'])

reDL = legal.RelationExtractionDLModel.pretrained("legre_notice_clause_xs", "en", "legal/models") \
    .setPredictionThreshold(0.1) \
    .setInputCols(["re_ner_chunks", "sentence"]) \
    .setOutputCol("relations")

pipeline = nlp.Pipeline(stages=[document_assembler,
                                sentence_detector, 
                                tokenizer,
                                pos_tagger,
                                dependency_parser, 
                                embeddings, 
                                ner_model, 
                                ner_converter,
                                re_filter, 
                                reDL])

empty_df = spark.createDataFrame([['']]).toDF("text")

re_model = pipeline.fit(empty_df)

light_model = nlp.LightPipeline(re_model)

text = """The addresses for notices shall be: IBM MSL 8501 IBM Drive 200 Baker Avenue Charlotte, NC 28262 Concord, MA 01742 Attn: MSL Project Office Attn: General Counsel  Telephone: 704-594-1964 Telephone: 978-287-5630 Facsimile: 704-594-4108 Facsimile: 978-287-5635  Either Party may change its address for this section by giving written notice to the other Party."""

result = light_model.fullAnnotate(text)

```

</div>

## Results

```bash
|   relation          |   entity1  |   entity1_begin  |   entity1_end  |   chunk1   |   entity2     |   entity2_begin  |   entity2_end  |   chunk2                                             |   confidence  |
|---------------------|------------|------------------|----------------|------------|---------------|------------------|----------------|------------------------------------------------------|---------------|
|   has_address       |   NAME     |   36             |   42           |   IBM MSL  |   ADDRESS     |   44             |   112          |   8501 IBM Drive 200 Baker Avenue Charlotte, NC ...  |   0.9997987   |
|   has_notice_party  |   NAME     |   36             |   42           |   IBM MSL  |   DEPARTMENT  |   120            |   137          |   MSL Project Office                                 |   0.34552842  |
|   has_title         |   NAME     |   36             |   42           |   IBM MSL  |   TITLE       |   145            |   159          |   General Counsel                                    |   0.48349348  |
|   has_phone         |   NAME     |   36             |   42           |   IBM MSL  |   PHONE       |   173            |   184          |   704-594-1964                                       |   0.99517375  |
|   has_phone         |   NAME     |   36             |   42           |   IBM MSL  |   PHONE       |   197            |   208          |   978-287-5630                                       |   0.9961247   |
|   has_fax           |   NAME     |   36             |   42           |   IBM MSL  |   FAX         |   221            |   232          |   704-594-4108                                       |   0.99340916  |
|   has_fax           |   NAME     |   36             |   42           |   IBM MSL  |   FAX         |   245            |   256          |   978-287-5635                                       |   0.97187006  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legre_notice_clause_xs|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|402.6 MB|

## References

In-house dataset

## Benchmarking

```bash

label             Recall  Precision  F1     Support 
has_address       0.976   1.000      0.988  41      
has_department    0.667   1.000      0.800  3       
has_email         1.000   1.000      1.000  7       
has_fax_phone     1.000   1.000      1.000  8       
has_notice_party  1.000   0.955      0.977  42      
has_person        1.000   0.938      0.968  15      
has_title         0.875   0.933      0.903  16      
other             1.000   1.000      1.000  68      
Avg.              0.940   0.978      0.954  -       
Weighted-Avg.     0.980   0.980      0.979  -  

```
