---
layout: model
title: Extract relations between drugs and proteins
author: John Snow Labs
name: re_drugprot_clinical
date: 2022-01-05
tags: [relation_extraction, clinical, en, licensed]
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 3.3.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect interactions between chemical compounds/drugs and genes/proteins using Spark NLP's `RelationExtractionModel()` by classifying whether a specified semantic relation holds between a chemical and gene entities within a sentence or document. The entity labels used during training were derived from the [custom NER model](https://nlp.johnsnowlabs.com/2021/12/20/ner_drugprot_clinical_en.html) created by our team for the [DrugProt corpus](https://zenodo.org/record/5119892). These include `CHEMICAL` for chemical compounds/drugs, `GENE` for genes/proteins and `GENE_AND_CHEMICAL` for entity mentions of type `GENE` and of type `CHEMICAL` that overlap (such as enzymes and small peptides). The relation categories from the [DrugProt corpus](https://zenodo.org/record/5119892) were condensed from 13 categories to 10 categories due to low numbers of examples for certain categories. This merging process involved grouping the `SUBSTRATE_PRODUCT-OF` and `SUBSTRATE` relation categories together and grouping the `AGONIST-ACTIVATOR`, `AGONIST-INHIBITOR` and `AGONIST` relation categories together.

## Predicted Entities

`INHIBITOR`, `DIRECT-REGULATOR`, `SUBSTRATE`, `ACTIVATOR`, `INDIRECT-UPREGULATOR`, `INDIRECT-DOWNREGULATOR`, `ANTAGONIST`, `PRODUCT-OF`, `PART-OF`, `AGONIST`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb#scrollTo=8tgB0NdZJlQU){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_drugprot_clinical_en_3.3.4_3.0_1641397921687.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")

words_embedder = WordEmbeddingsModel()\
    .pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

drugprot_ner_tagger = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")   

ner_converter = NerConverter()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

drugprot_re_model = RelationExtractionModel()\
        .pretrained("re_drugprot_clinical", "en", 'clinical/models')\
        .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(4)\
        .setPredictionThreshold(0.9)\ #default: 0.5 
        .setRelationPairs(['CHEMICAL-GENE']) # Possible relation pairs. Default: All Relations.

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, drugprot_ner_tagger, ner_converter, pos_tagger, dependency_parser, drugprot_re_model])

text='''Asparagine secretion by MSCs was directly related to their ASNS expression levels, suggesting a mechanism - increased concentrations of asparagine in the leukemic cell microenvironment - for the protective effects we observed.'''
data = spark.createDataFrame([[text]]).toDF("text")
result = pipeline.fit(data).transform(data)
```
```scala
...
val documenter = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val sentencer = SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = sparknlp.annotators.Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel()
    .pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val drugprot_ner_tagger = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags") 

val ner_converter = NerConverter()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models") 
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val drugprot_re_Model = RelationExactionModel()
    .pretrained("re_drugprot_clinical", "en", 'clinical/models')
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(4)
    .setPredictionThreshold(0.9) #default: 0.5 
    .setRelationPairs(Array("CHEMICAL-GENE")) # Possible relation pairs. Default: All Relations.

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, drugprot_ner_tagger, ner_converter, pos_tagger, dependency_parser, drugprot_re_Model))

val data = Seq("Asparagine secretion by MSCs was directly related to their ASNS expression levels, suggesting a mechanism - increased concentrations of asparagine in the leukemic cell microenvironment - for the protective effects we observed.").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
|  relation| entity1|entity1_begin|entity1_end|              chunk1|entity2|entity2_begin|entity2_end|              chunk2|confidence|
+----------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
|PRODUCT-OF|CHEMICAL|         1102|       1112|          Asparagine|   GENE|         1161|       1165|                ASNS|  0.998399|
|PRODUCT-OF|CHEMICAL|          160|        170|          asparagine|   GENE|          139|        143|                ASNS|  0.999304|
|ANTAGONIST|CHEMICAL|          613|        623|          aprepitant|   GENE|          580|        601|humanised SP rece...|  0.979057|
|ANTAGONIST|CHEMICAL|          625|        630|               EMEND|   GENE|          580|        601|humanised SP rece...|  0.981534|
| SUBSTRATE|CHEMICAL|          308|        310|                  PS|   GENE|          275|        283|            flippase|  0.991856|
| ACTIVATOR|CHEMICAL|         1563|       1578|     sn-1,2-glycerol|   GENE|         1479|       1509|plasma membrane P...|  0.988504|
| ACTIVATOR|CHEMICAL|         1563|       1578|     sn-1,2-glycerol|   GENE|         1511|       1517|              Atp8a1|  0.998399|
| ACTIVATOR|CHEMICAL|         2112|       2114|                  PE|   GENE|         2189|       2195|              Atp8a1|  0.994092|
| ACTIVATOR|CHEMICAL|         2116|       2145|phosphatidylhydro...|   GENE|         2189|       2195|              Atp8a1|  0.994409|
| ACTIVATOR|CHEMICAL|         2151|       2173|phosphatidylhomos...|   GENE|         2189|       2195|              Atp8a1|  0.981534|
+----------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_drugprot_clinical|
|Type:|re|
|Compatibility:|Spark NLP for Healthcare 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|9.7 MB|

## Data Source

This model was trained on the [DrugProt corpus](https://zenodo.org/record/5119892).

## Benchmarking

```bash
                        precision    recall  f1-score   support

             ACTIVATOR       0.39      0.29      0.33       235
               AGONIST       0.71      0.67      0.69       138
            ANTAGONIST       0.79      0.77      0.78       215
      DIRECT-REGULATOR       0.64      0.77      0.70       442
INDIRECT-DOWNREGULATOR       0.44      0.44      0.44       321
  INDIRECT-UPREGULATOR       0.49      0.43      0.46       292
             INHIBITOR       0.79      0.75      0.77      1119
               PART-OF       0.74      0.82      0.78       246
            PRODUCT-OF       0.51      0.37      0.43       153
             SUBSTRATE       0.58      0.69      0.63       486

              accuracy                           0.65      3647
             macro avg       0.61      0.60      0.60      3647
          weighted avg       0.65      0.65      0.64      3647
```