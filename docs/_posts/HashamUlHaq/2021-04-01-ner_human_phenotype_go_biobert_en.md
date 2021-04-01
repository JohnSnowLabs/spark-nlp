---
layout: model
title: Detect Normalized Genes and Human Phenotypes (biobert)
author: John Snow Labs
name: ner_human_phenotype_go_biobert
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model can be used to detect normalized mentions of genes (go) and human phenotypes (hp) in medical text.

## Predicted Entities

`HP`, `GO`

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_go_biobert_en_3.0.0_3.0_1617260627136.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("ner_human_phenotype_go_biobert", "en", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala

...
val embeddings_clinical = BertEmbeddings.pretrained("biobert_pubmed_base_cased")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("ner_human_phenotype_go_biobert", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val result = pipeline.fit(Seq.empty[""].toDS.toDF("text")).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_go_biobert|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|