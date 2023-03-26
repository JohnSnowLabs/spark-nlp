---
layout: model
title: Legal Arguments Mining in Court Decisions (in German)
author: John Snow Labs
name: legclf_argument_mining_german
date: 2023-03-26
tags: [de, licensed, classification, legal, tensorflow]
task: Text Classification
language: de
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: LegalClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multiclass classification model in German which classifies arguments in legal discourse. These are the following classes: `subsumption`, `definition`, `conclusion`, `other`.

## Predicted Entities

`subsumption`, `definition`, `conclusion`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_argument_mining_german_de_1.0.0_3.0_1679848514704.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_argument_mining_german_de_1.0.0_3.0_1679848514704.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
        .setInputCols(["document"])\
        .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_large_german_legal", "de")\
        .setInputCols(["document", "token"])\
        .setOutputCol("embeddings")\
        .setMaxSentenceLength(512)

embeddingsSentence = nlp.SentenceEmbeddings()\
        .setInputCols(["document", "embeddings"])\
        .setOutputCol("sentence_embeddings")\
        .setPoolingStrategy("AVERAGE")\


docClassifier = legal.ClassifierDLModel.pretrained("legclf_argument_mining_de", "de", "legal/models")\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCol("category")

nlpPipeline = nlp.Pipeline(stages=[
      documentAssembler, 
      tokenizer,
      embeddings,
      embeddingsSentence,
      docClassifier
])

df = spark.createDataFrame([["Folglich liegt eine Verletzung von Artikel 8 der Konvention vor ."]]).toDF("text")

model = nlpPipeline.fit(df)
result = model.transform(df)

result.select("text", "category.result").show(truncate=False)
```

</div>

## Results

```bash
+-----------------------------------------------------------------+------------+
|text                                                             |result      |
+-----------------------------------------------------------------+------------+
|Folglich liegt eine Verletzung von Artikel 8 der Konvention vor .|[conclusion]|
+-----------------------------------------------------------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_argument_mining_german|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|de|
|Size:|24.0 MB|

## References

Train dataset available [here](https://huggingface.co/datasets/MeilingShi/legal_argument_mining)

## Benchmarking

```bash
label         precision  recall    f1-score  support      
conclusion    0.88       0.88      0.88      52  
definition    0.83       0.83      0.83      58  
other         0.86       0.88      0.87      49  
subsumption   0.81       0.80      0.80      64  
accuracy         -          -      0.84      223                     
macro avg     0.85       0.85      0.85      223 
weighted avg  0.84       0.84      0.84      223 
```
