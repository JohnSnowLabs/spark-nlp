---
layout: model
title: Detect genes and human phenotypes
author: John Snow Labs
name: ner_human_phenotype_gene_clinical_en
date: 2020-09-21
tags: [ner, en, licensed]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Automatically detect mentions of genes and human phenotypes (hp) in medical text using Spark NLP for Healthcare pretrained models.

{:.h2_title}
## Predicted Entities 
GENE, HP

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_HUMAN_PHENOTYPE_GENE_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_en_2.5.5_2.4_1598558253840.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}


```python

clinical_ner = NerDLModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

nlpPipeline = Pipeline(stages=[clinical_ner])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

results = model.transform(data)

```

```scala

val ner = NerDLModel.pretrained("ner_human_phenotype_gene_clinical", "en", "clinical/models") \
  .setInputCols(["sentence", "token", "embeddings"]) \
  .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(ner))

val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(data)
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_gene_clinical_en|
|Type:|ner|
|Compatibility:|Spark NLP for Healthcare 2.6.0 +|
|Edition:|Healthcare|
|License:|Licensed|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and an "ner" column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select "token.result" and "ner.result" from your output dataframe or add the "Finisher" annotator to the end of your pipeline.
