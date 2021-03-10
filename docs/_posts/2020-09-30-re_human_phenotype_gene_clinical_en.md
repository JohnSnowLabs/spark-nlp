---
layout: model
title: Detect Relations Between Genes and Phenotypes
author: John Snow Labs
name: re_human_phenotype_gene_clinical
date: 2020-09-30
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.6.0
tags: [re, en, licensed, clinical]
supported: true
article_header:
    type: cover
use_language_switcher: "Python"
---

{:.h2_title}
## Description
This model can be used to identify relations between genes and phenotypes.

{:.h2_title}
## Included Relations
True-`1` : There is a relation between gene and phenotype.
False-`0` : There is not a relation between gene and phenotype.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_human_phenotype_gene_clinical_en_2.5.5_2.4_1598560152543.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_human_phenotype_gene_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["hp-gene",'gene-hp'])\
    .setMaxSyntacticDistance(4)
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos_tagger, dependecy_parser, word_embeddings, clinical_ner, ner_converter, clinical_re_Model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
annotations = light_pipeline.fullAnnotate("""Bilateral colobomatous microphthalmia and developmental delay in whom genetic studies identified a homozygous TENM3""")

```

```scala
...

val clinical_re_Model = RelationExtractionModel()
    .pretrained("re_human_phenotype_gene_clinical", "en", 'clinical/models')
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setRelationPairs(Array("hp-gene",'gene-hp'))
    .setMaxSyntacticDistance(4)

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos_tagger, dependecy_parser, word_embeddings, clinical_ner, ner_converter, clinical_re_Model))

val result = pipeline.fit(Seq.empty["Bilateral colobomatous microphthalmia and developmental delay in whom genetic studies identified a homozygous TENM3"].toDS.toDF("text")).transform(data)
```

</div>

{:.h2_title}
## Results

```bash
+----+------------+-----------+-----------------+---------------+---------------------+-----------+-----------------+---------------+---------------------+--------------+
|    |   relation | entity1   |   entity1_begin |   entity1_end | chunk1              | entity2   |   entity2_begin |   entity2_end | chunk2              |   confidence |
+====+============+===========+=================+===============+=====================+===========+=================+===============+=====================+==============+
|  0 |          1 | HP        |              23 |            36 | microphthalmia      | HP        |              42 |            60 | developmental delay |     0.999954 |
+----+------------+-----------+-----------------+---------------+---------------------+-----------+-----------------+---------------+---------------------+--------------+
|  1 |          1 | HP        |              23 |            36 | microphthalmia      | GENE      |             110 |           114 | TENM3               |     0.999999 |
+----+------------+-----------+-----------------+---------------+---------------------+-----------+-----------------+---------------+---------------------+--------------+
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_human_phenotype_gene_clinical|
|Type:|re|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[embeddings, pos_tags, ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|[en]|
|Case sensitive:|false|

## Data source
This model was trained with data from https://github.com/lasigeBioTM/PGR

For further details please refer to https://aclweb.org/anthology/papers/N/N19/N19-1152/