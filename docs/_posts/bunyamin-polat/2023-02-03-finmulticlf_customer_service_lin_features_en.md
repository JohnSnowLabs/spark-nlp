---
layout: model
title: Multilabel Classification of Customer Service (Linguistic features)
author: John Snow Labs
name: finmulticlf_customer_service_lin_features
date: 2023-02-03
tags: [en, licensed, finance, classification, customer, linguistic, tensorflow]
task: Text Classification
language: en
nav_key: models
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: MultiClassifierDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Multilabel Text Classification model that can help you classify a chat message from customer service according to linguistic features. The classes are the following:
 - Q - Colloquial variation
 - P - Politeness variation
 - W - Offensive language
 - K - Keyword language
 - B - Basic syntactic structure
 - C - Coordinated syntactic structure
 - I - Interrogative structure
 - M - Morphological variation (plurals, tenses…)
 - L - Lexical variation (synonyms)
 - E - Expanded abbreviations (I'm -> I am, I'd -> I would…)
 - N - Negation
 - Z - Noise phenomena like spelling or punctuation errors

## Predicted Entities

`B`, `C`, `E`, `I`, `K`, `L`, `M`, `N`, `P`, `Q`, `W`, `Z`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finmulticlf_customer_service_lin_features_en_1.0.0_3.0_1675430237309.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finmulticlf_customer_service_lin_features_en_1.0.0_3.0_1675430237309.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = nlp.UniversalSentenceEncoder.pretrained() \
    .setInputCols("document") \
    .setOutputCol("sentence_embeddings")

docClassifier = nlp.MultiClassifierDLModel().load("finmulticlf_customer_service_lin_features", "en", "finance/models")\
    .setInputCols("sentence_embeddings") \
    .setOutputCol("class")

pipeline = nlp.Pipeline().setStages(
      [
        document_assembler,
        embeddings,
        docClassifier
      ]
    )

empty_data = spark.createDataFrame([[""]]).toDF("text")
model = pipeline.fit(empty_data)
light_model = nlp.LightPipeline(model)

result = light_model.annotate("""What do i have to ddo to cancel a Gold account""")

result["class"]
```

</div>

## Results

```bash
['Q', 'B', 'L', 'Z', 'I']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finmulticlf_customer_service_lin_features|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|13.0 MB|

## References

https://github.com/bitext/customer-support-intent-detection-training-dataset

## Benchmarking

```bash
label         precision  recall  f1-score  support 
B             1.00       1.00    1.00      485     
C             0.79       0.80    0.80      61      
E             0.74       0.89    0.80      44      
I             0.95       0.94    0.94      134     
K             0.96       0.96    0.96      108     
L             0.96       0.97    0.96      402     
M             0.93       0.93    0.93      134     
N             0.90       0.75    0.82      12      
P             0.77       0.90    0.83      30      
Q             0.73       0.68    0.71      212     
W             0.85       0.88    0.87      33      
Z             0.68       0.72    0.70      160     
micro-avg     0.90       0.90    0.90      1815    
macro-avg     0.85       0.87    0.86      1815    
weighted-avg  0.90       0.90    0.90      1815    
samples-avg   0.91       0.92    0.90      1815   
```
