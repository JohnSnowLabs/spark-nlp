---
layout: model
title: Detect normalized genes and human phenotypes
author: John Snow Labs
name: ner_human_phenotype_go_clinical
date: 2020-09-21
tags: [ner, en, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model can be used to detect normalized mentions of genes (go) and human phenotypes (hp) in medical text.
## Predicted Entities: 
GO, HP

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GO_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_2.5.5_2.4_1598558253840.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use
Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python

clinical_ner = NerDLModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, clinical_ner, ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate("Another disease that shares two of the tumor components of CT, namely GIST and tricarboxylic acid cycle is the Carney-Stratakis syndrome (CSS) or dyad.")

```
</div>

{:.h2_title}
## Results

```bash
+----+--------------------------+---------+-------+----------+
|    | chunk                    |   begin |   end | entity   |
+====+==========================+=========+=======+==========+
|  0 | tumor                    |      39 |    43 | HP       |
+----+--------------------------+---------+-------+----------+
|  1 | tricarboxylic acid cycle |      79 |   102 | GO       |
+----+--------------------------+---------+-------+----------+
```
{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_go_clinical|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|Edition:|Official|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

