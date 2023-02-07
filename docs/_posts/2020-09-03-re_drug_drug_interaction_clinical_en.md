---
layout: model
title: Relation Extraction Model Clinical
author: John Snow Labs
name: re_drug_drug_interaction_clinical
class: RelationExtractionModel
language: en
repository: clinical/models
date: 2020-09-03
task: Relation Extraction
edition: Healthcare NLP 2.5.5
spark_version: 2.4
tags: [clinical,licensed,relation extraction,en]
supported: true
annotator: RelationExtractionModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Relation Extraction model based on syntactic features using deep learning. This model can be used to identify drug-drug interactions relationships among drug entities.

## Predicted Entities
``DDI-advise``, ``DDI-effect``, ``DDI-mechanism``, ``DDI-int``, ``DDI-false``

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_drug_drug_interaction_clinical_en_2.5.5_2.4_1599156924424.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_drug_drug_interaction_clinical_en_2.5.5_2.4_1599156924424.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}
{:.h2_title}
## How to use

In the table below, `re_drug_drug_interaction_clinical` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.

 |              RE MODEL             | RE MODEL LABES                                                       |   NER MODEL  | RE PAIRS      |
 |:---------------------------------:|-----------------------------------------------------------------------|:------------:|---------------|
 | re_drug_drug_interaction_clinical | DDI-advise,<br>DDI-effect,<br>DDI-mechanism,<br>DDI-int,<br>DDI-false | ner_posology | [“drug-drug”] |

 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
documenter = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")
  
words_embedder = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

ner_converter = NerConverter()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

ddi_re_model = RelationExtractionModel.pretrained("re_drug_drug_interaction_clinical","en","clinical/models")\
	.setInputCols("word_embeddings","chunk","pos","dependency")\
	.setOutputCol("category")
 
nlp_pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_converter, dependency_parser, ddi_re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

annotations = light_pipeline.fullAnnotate("""When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. If additional adrenergic drugs are to be administered by any route, they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated""")
```

```scala
val documenter = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentencer = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")
val words_embedder = WordEmbeddingsModel().pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val pos_tagger = PerceptronModel().pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val ner_tagger = MedicalNerModel.pretrained("ner_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_converter = new NerConverter() 
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel().pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val ddi_re_model = RelationExtractionModel.pretrained("re_drug_drug_interaction_clinical","en","clinical/models")
	.setInputCols("word_embeddings","chunk","pos","dependency")
	.setOutputCol("category")

val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_converter, dependency_parser, ddi_re_model))

val data = Seq("""When carbamazepine is withdrawn from the combination therapy, aripiprazole dose should then be reduced. If additional adrenergic drugs are to be administered by any route, they should be used with caution because the pharmacologically predictable sympathetic effects of Metformin may be potentiated""").toDS().toDF("text")

val result = pipeline.fit(data).transform(data)

```
</div>

{:.h2_title}
## Results

```bash

|relation   | entity1 | entity1_begin | entity1_end | chunk1         | entity2  |entity2_begin | entity2_end | chunk2        |
|DDI-advise | DRUG    |      5        |      17     | carbamazepine  |  DRUG    |     62             73      | aripiprazole  |

```
{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------------------|
| Name:           | re_drug_drug_interaction_clinical       |
| Type:    | RelationExtractionModel                 |
| Compatibility:  | Spark NLP 2.5.5+                                   |
| License:        | Licensed                                |
|Edition:|Official|                              |
|Input labels:         | [word_embeddings, chunk, pos, dependency] |
|Output labels:        | [category]                                |
| Language:       | en                                      |
| Case sensitive: | False                                   |
| Dependencies:  | embeddings_clinical                     |

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs.

{:.h2_title}
## Benchmarking
```bash
+-------------+------+------+------+
|     relation|recall| prec |   f1 |
+-------------+------+------+------+
|   DDI-effect|  0.76| 0.38 | 0.51 |
|    DDI-false|  0.72| 0.97 | 0.83 |
|   DDI-advise|  0.74| 0.39 | 0.51 |
+-------------+------+------+------+
```