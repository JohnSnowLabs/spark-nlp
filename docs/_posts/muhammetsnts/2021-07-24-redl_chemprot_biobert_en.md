---
layout: model
title: Extract relations between chemicals and proteins (ReDL)
author: John Snow Labs
name: redl_chemprot_biobert
date: 2021-07-24
tags: [relation_extraction, licensed, en, clinical]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.0.3
spark_version: 2.4
supported: true
annotator: RelationExtractionDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


Detect interactions between chemicals and proteins using BERT model by classifying whether a specified semantic relation holds between the chemical and protein entities within a sentence or document.


## Predicted Entities


`CPR:1`, `CPR:2`, `CPR:3`, `CPR:4`, `CPR:5`, `CPR:6`, `CPR:7`, `CPR:8`, `CPR:9`, `CPR:10`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CHEM_PROT){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/redl_chemprot_biobert_en_3.0.3_2.4_1627111978465.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use


In the table below, `redl_chemprot_biobert` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.


|        RE MODEL       |                             RE MODEL LABES                            |       NER MODEL       | RE PAIRS                  |
|:---------------------:|:---------------------------------------------------------------------:|:---------------------:|---------------------------|
| redl_chemprot_biobert | CPR:1, CPR:2, CPR:3, CPR:4, CPR:5, CPR:6, CPR:7, CPR:8, CPR:9, CPR:10 | ner_chemprot_clinical | [“No need to set pairs.”] |


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

tokenizer = sparknlp.annotators.Tokenizer()\
.setInputCols(["sentences"])\
.setOutputCol("tokens")

pos_tagger = PerceptronModel()\
.pretrained("pos_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"])\
.setOutputCol("pos_tags")

words_embedder = WordEmbeddingsModel() \
.pretrained("embeddings_clinical", "en", "clinical/models") \
.setInputCols(["sentences", "tokens"]) \
.setOutputCol("embeddings")

ner_tagger = MedicalNerModel.pretrained("ner_chemprot_clinical", "en", "clinical/models")\
.setInputCols("sentences", "tokens", "embeddings")\
.setOutputCol("ner_tags") 

ner_converter = NerConverter() \
.setInputCols(["sentences", "tokens", "ner_tags"]) \
.setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel() \
.pretrained("dependency_conllu", "en") \
.setInputCols(["sentences", "pos_tags", "tokens"]) \
.setOutputCol("dependencies")

# Set a filter on pairs of named entities which will be treated as relation candidates
re_ner_chunk_filter = RENerChunksFilter() \
.setInputCols(["ner_chunks", "dependencies"])\
.setMaxSyntacticDistance(10)\
.setOutputCol("re_ner_chunks")
#.setRelationPairs(['SYMPTOM-EXTERNAL_BODY_PART_OR_REGION'])

# The dataset this model is trained to is sentence-wise. 
# This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
re_model = RelationExtractionDLModel()\
.pretrained('redl_chemprot_biobert', 'en', "clinical/models") \
.setPredictionThreshold(0.5)\
.setInputCols(["re_ner_chunks", "sentences"]) \
.setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model])

text='''In this study, we examined the effects of mitiglinide on various cloned K(ATP) channels (Kir6.2/SUR1, Kir6.2/SUR2A, and Kir6.2/SUR2B) reconstituted in COS-1 cells, and compared them to another meglitinide-related compound, nateglinide. Patch-clamp analysis using inside-out recording configuration showed that mitiglinide inhibits the Kir6.2/SUR1 channel currents in a dose-dependent manner (IC50 value, 100 nM) but does not significantly inhibit either Kir6.2/SUR2A or Kir6.2/SUR2B channel currents even at high doses (more than 10 microM). Nateglinide inhibits Kir6.2/SUR1 and Kir6.2/SUR2B channels at 100 nM, and inhibits Kir6.2/SUR2A channels at high concentrations (1 microM). Binding experiments on mitiglinide, nateglinide, and repaglinide to SUR1 expressed in COS-1 cells revealed that they inhibit the binding of [3H]glibenclamide to SUR1 (IC50 values: mitiglinide, 280 nM; nateglinide, 8 microM; repaglinide, 1.6 microM), suggesting that they all share a glibenclamide binding site. The insulin responses to glucose, mitiglinide, tolbutamide, and glibenclamide in MIN6 cells after chronic mitiglinide, nateglinide, or repaglinide treatment were comparable to those after chronic tolbutamide and glibenclamide treatment. These results indicate that, similar to the sulfonylureas, mitiglinide is highly specific to the Kir6.2/SUR1 complex, i.e., the pancreatic beta-cell K(ATP) channel, and suggest that mitiglinide may be a clinically useful anti-diabetic drug.'''

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
...
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentencer = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentences")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentences"))
.setOutputCol("tokens")

val pos_tagger = PerceptronModel()
.pretrained("pos_clinical", "en", "clinical/models") 
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("pos_tags")

val words_embedder = WordEmbeddingsModel()
.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("embeddings")

val ner_tagger = MedicalNerModel.pretrained("ner_chemprot_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner_tags") 

val ner_converter = new NerConverter()
.setInputCols(Array("sentences", "tokens", "ner_tags"))
.setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
.pretrained("dependency_conllu", "en")
.setInputCols(Array("sentences", "pos_tags", "tokens"))
.setOutputCol("dependencies")

// Set a filter on pairs of named entities which will be treated as relation candidates
val re_ner_chunk_filter = RENerChunksFilter()
.setInputCols(Array("ner_chunks", "dependencies"))
.setMaxSyntacticDistance(10)
.setOutputCol("re_ner_chunks")
// .setRelationPairs(Array("SYMPTOM-EXTERNAL_BODY_PART_OR_REGION"))

// The dataset this model is trained to is sentence-wise. 
// This model can also be trained on document-level relations - in which case, while predicting, use "document" instead of "sentence" as input.
val re_model = RelationExtractionDLModel()
.pretrained("redl_chemprot_biobert", "en", "clinical/models")
.setPredictionThreshold(0.5)
.setInputCols(Array("re_ner_chunks", "sentences"))
.setOutputCol("relations")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, pos_tagger, words_embedder, ner_tagger, ner_converter, dependency_parser, re_ner_chunk_filter, re_model))

val data = Seq("In this study, we examined the effects of mitiglinide on various cloned K(ATP) channels (Kir6.2/SUR1, Kir6.2/SUR2A, and Kir6.2/SUR2B) reconstituted in COS-1 cells, and compared them to another meglitinide-related compound, nateglinide. Patch-clamp analysis using inside-out recording configuration showed that mitiglinide inhibits the Kir6.2/SUR1 channel currents in a dose-dependent manner (IC50 value, 100 nM) but does not significantly inhibit either Kir6.2/SUR2A or Kir6.2/SUR2B channel currents even at high doses (more than 10 microM). Nateglinide inhibits Kir6.2/SUR1 and Kir6.2/SUR2B channels at 100 nM, and inhibits Kir6.2/SUR2A channels at high concentrations (1 microM). Binding experiments on mitiglinide, nateglinide, and repaglinide to SUR1 expressed in COS-1 cells revealed that they inhibit the binding of [3H]glibenclamide to SUR1 (IC50 values: mitiglinide, 280 nM; nateglinide, 8 microM; repaglinide, 1.6 microM), suggesting that they all share a glibenclamide binding site. The insulin responses to glucose, mitiglinide, tolbutamide, and glibenclamide in MIN6 cells after chronic mitiglinide, nateglinide, or repaglinide treatment were comparable to those after chronic tolbutamide and glibenclamide treatment. These results indicate that, similar to the sulfonylureas, mitiglinide is highly specific to the Kir6.2/SUR1 complex, i.e., the pancreatic beta-cell K(ATP) channel, and suggest that mitiglinide may be a clinically useful anti-diabetic drug.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.chemprot").predict("""In this study, we examined the effects of mitiglinide on various cloned K(ATP) channels (Kir6.2/SUR1, Kir6.2/SUR2A, and Kir6.2/SUR2B) reconstituted in COS-1 cells, and compared them to another meglitinide-related compound, nateglinide. Patch-clamp analysis using inside-out recording configuration showed that mitiglinide inhibits the Kir6.2/SUR1 channel currents in a dose-dependent manner (IC50 value, 100 nM) but does not significantly inhibit either Kir6.2/SUR2A or Kir6.2/SUR2B channel currents even at high doses (more than 10 microM). Nateglinide inhibits Kir6.2/SUR1 and Kir6.2/SUR2B channels at 100 nM, and inhibits Kir6.2/SUR2A channels at high concentrations (1 microM). Binding experiments on mitiglinide, nateglinide, and repaglinide to SUR1 expressed in COS-1 cells revealed that they inhibit the binding of [3H]glibenclamide to SUR1 (IC50 values: mitiglinide, 280 nM; nateglinide, 8 microM; repaglinide, 1.6 microM), suggesting that they all share a glibenclamide binding site. The insulin responses to glucose, mitiglinide, tolbutamide, and glibenclamide in MIN6 cells after chronic mitiglinide, nateglinide, or repaglinide treatment were comparable to those after chronic tolbutamide and glibenclamide treatment. These results indicate that, similar to the sulfonylureas, mitiglinide is highly specific to the Kir6.2/SUR1 complex, i.e., the pancreatic beta-cell K(ATP) channel, and suggest that mitiglinide may be a clinically useful anti-diabetic drug.""")
```

</div>


## Results


```bash
|    | relation   | entity1   |   entity1_begin |   entity1_end | chunk1            | entity2   |   entity2_begin |   entity2_end | chunk2        |   confidence |
|---:|:-----------|:----------|----------------:|--------------:|:------------------|:----------|----------------:|--------------:|:--------------|-------------:|
|  0 | CPR:2      | CHEMICAL  |              43 |            53 | mitiglinide       | GENE-N    |              80 |            87 | channels      |     0.998399 |
|  1 | CPR:2      | GENE-N    |              80 |            87 | channels          | CHEMICAL  |             224 |           234 | nateglinide   |     0.994489 |
|  2 | CPR:2      | CHEMICAL  |             706 |           716 | mitiglinide       | GENE-Y    |             751 |           754 | SUR1          |     0.999304 |
|  3 | CPR:2      | CHEMICAL  |             823 |           839 | [3H]glibenclamide | GENE-Y    |             844 |           847 | SUR1          |     0.998923 |
|  4 | CPR:2      | GENE-N    |             998 |          1004 | insulin           | CHEMICAL  |            1019 |          1025 | glucose       |     0.979057 |
|  5 | CPR:2      | GENE-N    |             998 |          1004 | insulin           | CHEMICAL  |            1028 |          1038 | mitiglinide   |     0.988504 |
|  6 | CPR:2      | GENE-N    |             998 |          1004 | insulin           | CHEMICAL  |            1041 |          1051 | tolbutamide   |     0.991856 |
|  7 | CPR:2      | GENE-N    |             998 |          1004 | insulin           | CHEMICAL  |            1058 |          1070 | glibenclamide |     0.994092 |
|  8 | CPR:2      | GENE-N    |             998 |          1004 | insulin           | CHEMICAL  |            1100 |          1110 | mitiglinide   |     0.994409 |
|  9 | CPR:2      | CHEMICAL  |            1290 |          1300 | mitiglinide       | GENE-N    |            1387 |          1393 | channel       |     0.981534 |
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|redl_chemprot_biobert|
|Compatibility:|Healthcare NLP 3.0.3+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Case sensitive:|true|


## Data Source


Trained on ChemProt benchmark dataset.


## Benchmarking


```bash
Relation           Recall Precision        F1   Support
CPR:1               0.870     0.908     0.888       215
CPR:10              0.818     0.762     0.789       258
CPR:2               0.726     0.806     0.764      1651
CPR:3               0.788     0.785     0.787       657
CPR:4               0.901     0.855     0.878      1599
CPR:5               0.799     0.891     0.842       184
CPR:6               0.888     0.845     0.866       258
CPR:7               0.520     0.765     0.619        25
CPR:8               0.083     0.333     0.133        24
CPR:9               0.930     0.805     0.863       629
Avg.                0.732     0.775     0.743		-
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE1NTU4MzY3MV19
-->