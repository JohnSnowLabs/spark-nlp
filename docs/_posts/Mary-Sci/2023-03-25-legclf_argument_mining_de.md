---
layout: model
title: Legal Arguments Mining in Court Decisions
author: John Snow Labs
name: legclf_argument_mining
date: 2023-03-25
tags: [licensed, en, classification, legal, de, tensorflow]
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

This is a Multiclass classification model which classifies arguments in legal discourse. These are the following classes: `subsumption`, `definition`, `conclusion`, `other`.

## Predicted Entities

`subsumption`, `definition`, `conclusion`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_argument_mining_de_1.0.0_3.0_1679760684128.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_argument_mining_de_1.0.0_3.0_1679760684128.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en")\
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

embeddingsSentence = (
    nlp.SentenceEmbeddings()
    .setInputCols(["document", "embeddings"])
    .setOutputCol("sentence_embeddings")
    .setPoolingStrategy("AVERAGE")
)

docClassifier = legal.ClassifierDLModel.load("legclf_argument_mining","en", "legal/models")\
      .setInputCols(["sentence_embeddings"])\
      .setOutputCol("category")

nlpPipeline = nlp.Pipeline(stages=[
      documentAssembler, 
      tokenizer,
      embeddings,
      embeddingsSentence,
      docClassifier
])

df = spark.createDataFrame([["There is therefore no doubt – and the Government do not contest – that the measures concerned in the present case ( the children 's continued placement in foster homes and the restrictions imposed on contact between the applicants and their children ) amounts to an “ interference ” with the applicants ' rights to respect for their family life ."]]).toDF("text")

model = nlpPipeline.fit(df)

result = model.transform(df)

result.select("text", "category.result").show()
```

</div>

## Results

```bash
+--------------------+-------------+
|                text|       result|
+--------------------+-------------+
|There is therefor...|[subsumption]|
+--------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_argument_mining|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|de|
|Size:|22.2 MB|

## References

Train dataset available [here](https://huggingface.co/datasets/MeilingShi/legal_argument_mining)

## Benchmarking

```bash
 label         precision    recall    f1-score  support      
 conclusion    0.93         0.79      0.85      52  
 definition    0.87         0.81      0.84      58  
 other         0.88         0.88      0.88      57  
 subsumption   0.64         0.79      0.71      52  
 accuracy      0.82         219          -       -  
 macro-avg     0.83         0.82      0.82      219 
 weighted-avg  0.83         0.82      0.82      219 
```
