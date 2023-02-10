---
layout: model
title: Extract relations between drugs and proteins
author: John Snow Labs
name: re_drugprot_clinical
date: 2022-01-05
tags: [relation_extraction, clinical, en, licensed]
task: Relation Extraction
language: en
edition: Healthcare NLP 3.3.4
spark_version: 3.0
supported: true
annotator: RelationExtractionModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


NOTE: This model has been improved by a new SOTA, Bert-based, Relation Extraction model, you can find [here](https://nlp.johnsnowlabs.com/2022/01/05/redl_drugprot_biobert_en.html)


Detect interactions between chemical compounds/drugs and genes/proteins using Spark NLP's `RelationExtractionModel()` by classifying whether a specified semantic relation holds between a chemical and gene entities within a sentence or document. The entity labels used during training were derived from the [custom NER model](https://nlp.johnsnowlabs.com/2021/12/20/ner_drugprot_clinical_en.html) created by our team for the [DrugProt corpus](https://zenodo.org/record/5119892). These include `CHEMICAL` for chemical compounds/drugs, `GENE` for genes/proteins and `GENE_AND_CHEMICAL` for entity mentions of type `GENE` and of type `CHEMICAL` that overlap (such as enzymes and small peptides). The relation categories from the [DrugProt corpus](https://zenodo.org/record/5119892) were condensed from 13 categories to 10 categories due to low numbers of examples for certain categories. This merging process involved grouping the `SUBSTRATE_PRODUCT-OF` and `SUBSTRATE` relation categories together and grouping the `AGONIST-ACTIVATOR`, `AGONIST-INHIBITOR` and `AGONIST` relation categories together.


## Predicted Entities


`INHIBITOR`, `DIRECT-REGULATOR`, `SUBSTRATE`, `ACTIVATOR`, `INDIRECT-UPREGULATOR`, `INDIRECT-DOWNREGULATOR`, `ANTAGONIST`, `PRODUCT-OF`, `PART-OF`, `AGONIST`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb#scrollTo=8tgB0NdZJlQU){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_drugprot_clinical_en_3.3.4_3.0_1641397921687.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_drugprot_clinical_en_3.3.4_3.0_1641397921687.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use


In the table below, `re_drugprot_clinical` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.




|       RE MODEL       |                                                                                 RE MODEL LABES                                                                                |       NER MODEL       | RE PAIRS                                                                               |
|:--------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------:|----------------------------------------------------------------------------------------|
| re_drugprot_clinical | INHIBITOR, <br>DIRECT-REGULATOR, <br>SUBSTRATE, <br>ACTIVATOR, <br>INDIRECT-UPREGULATOR, <br>INDIRECT-DOWNREGULATOR, <br>ANTAGONIST, <br>PRODUCT-OF, <br>PART-OF, <br>AGONIST | ner_drugprot_clinical | [“checmical-gene”, <br>“chemical-gene_and_chemical”, <br>“gene_and_chemical-gene”]<br> |




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
.setPredictionThreshold(0.9)\
.setRelationPairs(['CHEMICAL-GENE']) # Possible relation pairs. Default: All Relations.

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, drugprot_ner_tagger, ner_converter, pos_tagger, dependency_parser, drugprot_re_model])

text='''Lipid specific activation of the murine P4-ATPase Atp8a1 (ATPase II). The asymmetric transbilayer distribution of phosphatidylserine (PS) in the mammalian plasma membrane and secretory vesicles is maintained, in part, by an ATP-dependent transporter. This aminophospholipid "flippase" selectively transports PS to the cytosolic leaflet of the bilayer and is sensitive to vanadate, Ca(2+), and modification by sulfhydryl reagents. Although the flippase has not been positively identified, a subfamily of P-type ATPases has been proposed to function as transporters of amphipaths, including PS and other phospholipids. A candidate PS flippase ATP8A1 (ATPase II), originally isolated from bovine secretory vesicles, is a member of this subfamily based on sequence homology to the founding member of the subfamily, the yeast protein Drs2, which has been linked to ribosomal assembly, the formation of Golgi-coated vesicles, and the maintenance of PS asymmetry. To determine if ATP8A1 has biochemical characteristics consistent with a PS flippase, a murine homologue of this enzyme was expressed in insect cells and purified. The purified Atp8a1 is inactive in detergent micelles or in micelles containing phosphatidylcholine, phosphatidic acid, or phosphatidylinositol, is minimally activated by phosphatidylglycerol or phosphatidylethanolamine (PE), and is maximally activated by PS. The selectivity for PS is dependent upon multiple elements of the lipid structure. Similar to the plasma membrane PS transporter, Atp8a1 is activated only by the naturally occurring sn-1,2-glycerol isomer of PS and not the sn-2,3-glycerol stereoisomer. Both flippase and Atp8a1 activities are insensitive to the stereochemistry of the serine headgroup. Most modifications of the PS headgroup structure decrease recognition by the plasma membrane PS flippase. Activation of Atp8a1 is also reduced by these modifications; phosphatidylserine-O-methyl ester, lysophosphatidylserine, glycerophosphoserine, and phosphoserine, which are not transported by the plasma membrane flippase, do not activate Atp8a1. Weakly translocated lipids (PE, phosphatidylhydroxypropionate, and phosphatidylhomoserine) are also weak Atp8a1 activators. However, N-methyl-phosphatidylserine, which is transported by the plasma membrane flippase at a rate equivalent to PS, is incapable of activating Atp8a1 activity. These results indicate that the ATPase activity of the secretory granule Atp8a1 is activated by phospholipids binding to a specific site whose properties (PS selectivity, dependence upon glycerol but not serine, stereochemistry, and vanadate sensitivity) are similar to, but distinct from, the properties of the substrate binding site of the plasma membrane flippase.'''

data = spark.createDataFrame([[text]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
...
val documenter = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentencer = new SentenceDetector()
.setInputCols("document")
.setOutputCol("sentences")

val tokenizer = new Tokenizer()
.setInputCols("sentences")
.setOutputCol("tokens")

val words_embedder = WordEmbeddingsModel()
.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens"))
.setOutputCol("embeddings")

val drugprot_ner_tagger = MedicalNerModel.pretrained("ner_drugprot_clinical", "en", "clinical/models")
.setInputCols(Array("sentences", "tokens", "embeddings"))
.setOutputCol("ner_tags") 

val ner_converter = new NerConverter()
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
.pretrained("re_drugprot_clinical", "en", "clinical/models")
.setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
.setOutputCol("relations")
.setMaxSyntacticDistance(4)
.setPredictionThreshold(0.9)
.setRelationPairs(Array("CHEMICAL-GENE")) # Possible relation pairs. Default: All Relations.

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, drugprot_ner_tagger, ner_converter, pos_tagger, dependency_parser, drugprot_re_Model))

val data = Seq("""Lipid specific activation of the murine P4-ATPase Atp8a1 (ATPase II). The asymmetric transbilayer distribution of phosphatidylserine (PS) in the mammalian plasma membrane and secretory vesicles is maintained, in part, by an ATP-dependent transporter. This aminophospholipid "flippase" selectively transports PS to the cytosolic leaflet of the bilayer and is sensitive to vanadate, Ca(2+), and modification by sulfhydryl reagents. Although the flippase has not been positively identified, a subfamily of P-type ATPases has been proposed to function as transporters of amphipaths, including PS and other phospholipids. A candidate PS flippase ATP8A1 (ATPase II), originally isolated from bovine secretory vesicles, is a member of this subfamily based on sequence homology to the founding member of the subfamily, the yeast protein Drs2, which has been linked to ribosomal assembly, the formation of Golgi-coated vesicles, and the maintenance of PS asymmetry. To determine if ATP8A1 has biochemical characteristics consistent with a PS flippase, a murine homologue of this enzyme was expressed in insect cells and purified. The purified Atp8a1 is inactive in detergent micelles or in micelles containing phosphatidylcholine, phosphatidic acid, or phosphatidylinositol, is minimally activated by phosphatidylglycerol or phosphatidylethanolamine (PE), and is maximally activated by PS. The selectivity for PS is dependent upon multiple elements of the lipid structure. Similar to the plasma membrane PS transporter, Atp8a1 is activated only by the naturally occurring sn-1,2-glycerol isomer of PS and not the sn-2,3-glycerol stereoisomer. Both flippase and Atp8a1 activities are insensitive to the stereochemistry of the serine headgroup. Most modifications of the PS headgroup structure decrease recognition by the plasma membrane PS flippase. Activation of Atp8a1 is also reduced by these modifications; phosphatidylserine-O-methyl ester, lysophosphatidylserine, glycerophosphoserine, and phosphoserine, which are not transported by the plasma membrane flippase, do not activate Atp8a1. Weakly translocated lipids (PE, phosphatidylhydroxypropionate, and phosphatidylhomoserine) are also weak Atp8a1 activators. However, N-methyl-phosphatidylserine, which is transported by the plasma membrane flippase at a rate equivalent to PS, is incapable of activating Atp8a1 activity. These results indicate that the ATPase activity of the secretory granule Atp8a1 is activated by phospholipids binding to a specific site whose properties (PS selectivity, dependence upon glycerol but not serine, stereochemistry, and vanadate sensitivity) are similar to, but distinct from, the properties of the substrate binding site of the plasma membrane flippase.""").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.relation.drugprot.clinical").predict("""Lipid specific activation of the murine P4-ATPase Atp8a1 (ATPase II). The asymmetric transbilayer distribution of phosphatidylserine (PS) in the mammalian plasma membrane and secretory vesicles is maintained, in part, by an ATP-dependent transporter. This aminophospholipid "flippase" selectively transports PS to the cytosolic leaflet of the bilayer and is sensitive to vanadate, Ca(2+), and modification by sulfhydryl reagents. Although the flippase has not been positively identified, a subfamily of P-type ATPases has been proposed to function as transporters of amphipaths, including PS and other phospholipids. A candidate PS flippase ATP8A1 (ATPase II), originally isolated from bovine secretory vesicles, is a member of this subfamily based on sequence homology to the founding member of the subfamily, the yeast protein Drs2, which has been linked to ribosomal assembly, the formation of Golgi-coated vesicles, and the maintenance of PS asymmetry. To determine if ATP8A1 has biochemical characteristics consistent with a PS flippase, a murine homologue of this enzyme was expressed in insect cells and purified. The purified Atp8a1 is inactive in detergent micelles or in micelles containing phosphatidylcholine, phosphatidic acid, or phosphatidylinositol, is minimally activated by phosphatidylglycerol or phosphatidylethanolamine (PE), and is maximally activated by PS. The selectivity for PS is dependent upon multiple elements of the lipid structure. Similar to the plasma membrane PS transporter, Atp8a1 is activated only by the naturally occurring sn-1,2-glycerol isomer of PS and not the sn-2,3-glycerol stereoisomer. Both flippase and Atp8a1 activities are insensitive to the stereochemistry of the serine headgroup. Most modifications of the PS headgroup structure decrease recognition by the plasma membrane PS flippase. Activation of Atp8a1 is also reduced by these modifications; phosphatidylserine-O-methyl ester, lysophosphatidylserine, glycerophosphoserine, and phosphoserine, which are not transported by the plasma membrane flippase, do not activate Atp8a1. Weakly translocated lipids (PE, phosphatidylhydroxypropionate, and phosphatidylhomoserine) are also weak Atp8a1 activators. However, N-methyl-phosphatidylserine, which is transported by the plasma membrane flippase at a rate equivalent to PS, is incapable of activating Atp8a1 activity. These results indicate that the ATPase activity of the secretory granule Atp8a1 is activated by phospholipids binding to a specific site whose properties (PS selectivity, dependence upon glycerol but not serine, stereochemistry, and vanadate sensitivity) are similar to, but distinct from, the properties of the substrate binding site of the plasma membrane flippase.""")
```

</div>


## Results


```bash
+---------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
| relation| entity1|entity1_begin|entity1_end|              chunk1|entity2|entity2_begin|entity2_end|              chunk2|confidence|
+---------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
|SUBSTRATE|CHEMICAL|          308|        310|                  PS|   GENE|          275|        283|            flippase|  0.998399|
|ACTIVATOR|CHEMICAL|         1563|       1578|     sn-1,2-glycerol|   GENE|         1479|       1509|plasma membrane P...|  0.999304|
|ACTIVATOR|CHEMICAL|         1563|       1578|     sn-1,2-glycerol|   GENE|         1511|       1517|              Atp8a1|  0.979057|
|ACTIVATOR|CHEMICAL|         2112|       2114|                  PE|   GENE|         2189|       2195|              Atp8a1|  0.998299|
|ACTIVATOR|CHEMICAL|         2116|       2145|phosphatidylhydro...|   GENE|         2189|       2195|              Atp8a1|  0.981534|
|ACTIVATOR|CHEMICAL|         2151|       2173|phosphatidylhomos...|   GENE|         2189|       2195|              Atp8a1|  0.988504|
|SUBSTRATE|CHEMICAL|         2217|       2244|N-methyl-phosphat...|   GENE|         2290|       2298|            flippase|  0.994092|
|ACTIVATOR|CHEMICAL|         1292|       1312|phosphatidylglycerol|   GENE|         1134|       1140|              Atp8a1|  0.994409|
|ACTIVATOR|CHEMICAL|         1316|       1340|phosphatidylethan...|   GENE|         1134|       1140|              Atp8a1|  0.988359|
|ACTIVATOR|CHEMICAL|         1342|       1344|                  PE|   GENE|         1134|       1140|              Atp8a1|  0.988399|
|ACTIVATOR|CHEMICAL|         1377|       1379|                  PS|   GENE|         1134|       1140|              Atp8a1|  0.996349|
|ACTIVATOR|CHEMICAL|         2526|       2528|                  PS|   GENE|         2444|       2450|              Atp8a1|  0.978597|
|ACTIVATOR|CHEMICAL|         2526|       2528|                  PS|   GENE|         2403|       2409|              ATPase|  0.988679|
+---------+--------+-------------+-----------+--------------------+-------+-------------+-----------+--------------------+----------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|re_drugprot_clinical|
|Type:|re|
|Compatibility:|Healthcare NLP 3.3.4+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Size:|9.7 MB|


## Data Source


This model was trained on the [DrugProt corpus](https://zenodo.org/record/5119892).

This model has been improved using a Deep Learning Relation Extraction approach, resulting in the model available [here](https://nlp.johnsnowlabs.com/2022/01/05/redl_drugprot_biobert_en.html) with the following metrics

## Benchmarking


```bash
label              precision     recall  f1-score    support
ACTIVATOR               0.39       0.29      0.33        235
AGONIST                 0.71       0.67      0.69        138
ANTAGONIST              0.79       0.77      0.78        215
DIRECT-REGULATOR        0.64       0.77      0.70        442
INDIRECT-DOWNREGULATOR  0.44       0.44      0.44        321
INDIRECT-UPREGULATOR    0.49       0.43      0.46        292
INHIBITOR               0.79       0.75      0.77       1119
PART-OF                 0.74       0.82      0.78        246
PRODUCT-OF              0.51       0.37      0.43        153
SUBSTRATE               0.58       0.69      0.63        486
accuracy                  -          -       0.65       3647
macro-avg               0.61       0.60      0.60       3647
weighted-avg            0.65       0.65      0.64       3647
-                       -           -       -           -
ACTIVATOR               0.885      0.776     0.827       235
AGONIST                 0.810      0.925     0.864       137
ANTAGONIST              0.970      0.919     0.944       199
DIRECT-REGULATOR        0.836      0.901     0.867       403
INDIRECT-DOWNREGULATOR  0.885      0.850     0.867       313
INDIRECT-UPREGULATOR    0.844      0.887     0.865       270
INHIBITOR               0.947      0.937     0.942       1083
PART-OF                 0.939      0.889     0.913       247
PRODUCT-OF              0.697      0.953     0.805       145
SUBSTRATE               0.912      0.884     0.898       468
Avg                     0.873      0.892     0.879       3647
Weighted-Avg            0.897      0.899     0.897       3647
```
