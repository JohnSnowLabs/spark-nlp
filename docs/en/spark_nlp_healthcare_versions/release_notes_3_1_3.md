---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.1.3
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_1_3
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.1.3
We are glad to announce that Spark NLP for Healthcare 3.1.3 has been released!.
This release comes with new features, new models, bug fixes, and examples.

#### Highlights
+ New Relation Extraction model and a Pretrained pipeline for extracting and linking ADEs
+ New Entity Resolver model for SNOMED codes
+ ChunkConverter Annotator
+ BugFix: getAnchorDateMonth method in DateNormalizer.
+ BugFix: character map in MedicalNerModel fine-tuning.

##### New Relation Extraction model and a Pretrained pipeline for extracting and linking ADEs

We are releasing a new Relation Extraction Model for ADEs. This model is trained using Bert Word embeddings (`biobert_pubmed_base_cased`), and is capable of linking ADEs and Drugs.

*Example*:

```python
re_model = RelationExtractionModel()\
        .pretrained("re_ade_biobert", "en", 'clinical/models')\
        .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(3)\ #default: 0
        .setPredictionThreshold(0.5)\ #default: 0.5
        .setRelationPairs(["ade-drug", "drug-ade"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

text ="""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps"""

annotations = light_pipeline.fullAnnotate(text)
```

We also have a new pipeline comprising of all models related to ADE(Adversal Drug Event) as part of this release. This pipeline includes classification, NER, assertion and relation extraction models. Users can now use this pipeline to get classification result, ADE and Drug entities, assertion status for ADE entities, and relations between ADE and Drug entities.

Example:

```python
    pretrained_ade_pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

    result = pretrained_ade_pipeline.fullAnnotate("""Been taking Lipitor for 15 years , have experienced sever fatigue a lot!!! . Doctor moved me to voltaren 2 months ago , so far , have only experienced cramps""")[0]
```

*Results*:

```bash
Class: True

NER_Assertion:
|    | chunk                   | entitiy    | assertion   |
|----|-------------------------|------------|-------------|
| 0  | Lipitor                 | DRUG       | -           |
| 1  | sever fatigue           | ADE        | Conditional |
| 2  | voltaren                | DRUG       | -           |
| 3  | cramps                  | ADE        | Conditional |

Relations:
|    | chunk1                        | entitiy1   | chunk2      | entity2 | relation |
|----|-------------------------------|------------|-------------|---------|----------|
| 0  | sever fatigue                 | ADE        | Lipitor     | DRUG    |        1 |
| 1  | cramps                        | ADE        | Lipitor     | DRUG    |        0 |
| 2  | sever fatigue                 | ADE        | voltaren    | DRUG    |        0 |
| 3  | cramps                        | ADE        | voltaren    | DRUG    |        1 |

```
##### New Entity Resolver model for SNOMED codes

We are releasing a new SentenceEntityResolver model for SNOMED codes. This model also includes AUX SNOMED concepts and can find codes for Morph Abnormality, Procedure, Substance, Physical Object, and Body Structure entities. In the metadata, the `all_k_aux_labels` can be divided to get further information: `ground truth`, `concept`, and `aux` . In the example shared below the ground truth is `Atherosclerosis`, concept is `Observation`, and aux is `Morph Abnormality`.

*Example*:

```python
snomed_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_snomed_findings_aux_concepts", "en", "clinical/models") \
     .setInputCols(["ner_chunk", "sbert_embeddings"]) \
     .setOutputCol("snomed_code")\
     .setDistanceFunction("EUCLIDEAN")

snomed_pipelineModel = PipelineModel(
    stages = [
        documentAssembler,
        sbert_embedder,
        snomed_resolver])

snomed_lp = LightPipeline(snomed_pipelineModel)
result = snomed_lp.fullAnnotate("atherosclerosis")

```

*Results*:

```bash
|    | chunks          | code     | resolutions                                                                                                                                                                                                                                                                                                                                                                                                                    | all_codes                                                                                                                                                                                          | all_k_aux_labels                                      | all_distances                                                                                                                                   |
|---:|:----------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | atherosclerosis | 38716007 | [atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis, atherosclerosis artery, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, coronary atherosclerosis, arteriosclerosis, carotid atherosclerosis, cardiovascular arteriosclerosis, aortic atherosclerosis, aortic atherosclerosis, atherosclerotic ischemic disease] | [38716007, 155382007, 155414001, 195251000, 266318005, 194848007, 441574008, 443502000, 41702007, 266231003, 155316000, 194841001, 28960008, 300920004, 39468009, 155415000, 195252007, 129573006] | 'Atherosclerosis', 'Observation', 'Morph Abnormality' | [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0280, 0.0451, 0.0451, 0.0451, 0.0451, 0.0451, 0.0462, 0.0477, 0.0466, 0.0490, 0.0490, 0.0485 |
```


##### ChunkConverter Annotator

Allows to use RegexMather chunks as NER chunks and feed the output to the downstream annotators like RE or Deidentification.

``` Python
        document_assembler = DocumentAssembler().setInputCol('text').setOutputCol('document')

        sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")

        regex_matcher = RegexMatcher()\
            .setInputCols("sentence")\
            .setOutputCol("regex")\
            .setExternalRules(path="../src/test/resources/regex-matcher/rules.txt",delimiter=",")

        chunkConverter = ChunkConverter().setInputCols("regex").setOutputCol("chunk")


```

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_1_2">Version 3.1.2</a>
    </li>
    <li>
        <strong>Version 3.1.3</strong>
    </li>
    <li>
        <a href="release_notes_3_2_0">Version 3.2.0</a>
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
    <li class="active"><a href="release_notes_3_1_3">3.1.3</a></li>
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
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
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