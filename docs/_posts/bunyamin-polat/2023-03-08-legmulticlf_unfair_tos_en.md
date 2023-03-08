---
layout: model
title: Legal Multilabel Classification on Terms of Service (UNFAIR-ToS)
author: John Snow Labs
name: legmulticlf_unfair_tos
date: 2023-03-08
tags: [en, legal, licensed, classification, unfair, tensorflow]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel Text Classification model that can help you classify 8 types of unfair contractual terms (sentences), meaning terms that potentially violate user rights according to European consumer law.

## Predicted Entities

`Arbitration`, `Choice_of_Law`, `Content_Removal`, `Contract_by_Using`, `Jurisdiction`, `Limitation_of_Liability`, `Unilateral_Change`, `Unilateral_Termination`, `Other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legmulticlf_unfair_tos_en_1.0.0_3.0_1678283272065.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legmulticlf_unfair_tos_en_1.0.0_3.0_1678283272065.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

embeddings = nlp.RoBertaEmbeddings.pretrained("roberta_embeddings_legal_roberta_base", "en")\
    .setInputCols(["document", "token"])\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)

embeddingsSentence = nlp.SentenceEmbeddings()\
    .setInputCols(["document", "embeddings"])\
    .setOutputCol("sentence_embeddings")\
    .setPoolingStrategy("AVERAGE")

docClassifier = nlp.MultiClassifierDLModel().pretrained("legmulticlf_unfair_tos", "en", "legal/models")\
    .setInputCols("sentence_embeddings") \
    .setOutputCol("class")

pipeline = nlp.Pipeline(
    stages=[
        document_assembler,
        tokenizer,
        embeddings,
        embeddingsSentence,
        docClassifier
    ]
)

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

light_model = nlp.LightPipeline(model)

result = light_model.annotate("""we may alter, suspend or discontinue any aspect of the service at any time, including the availability of any service feature, database or content.""")

```

</div>

## Results

```bash
['Unilateral_Change', 'Unilateral_Termination']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legmulticlf_unfair_tos|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|13.9 MB|

## References

Train dataset available [here](https://github.com/coastalcph/lex-glue)

## Benchmarking

```bash
label                    precision  recall  f1-score  support 
Arbitration              1.00       0.82    0.90      11      
Choice_of_Law            0.93       0.93    0.93      14      
Content_Removal          0.80       0.57    0.67      21      
Contract_by_Using        0.93       0.82    0.87      17      
Jurisdiction             1.00       1.00    1.00      16      
Limitation_of_Liability  0.81       0.80    0.81      60      
Other                    0.78       0.71    0.75      66      
Unilateral_Change        0.94       0.84    0.89      38      
Unilateral_Termination   0.78       0.81    0.79      36      
micro-avg                0.85       0.79    0.82      279     
macro-avg                0.89       0.81    0.85      279     
weighted-avg             0.85       0.79    0.82      279     
samples-avg              0.78       0.80    0.78      279 
```
