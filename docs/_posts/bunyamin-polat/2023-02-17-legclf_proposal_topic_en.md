---
layout: model
title: Legal Proposal Classification
author: John Snow Labs
name: legclf_proposal_topic
date: 2023-02-17
tags: [en, legal, classification, proposal, licensed, tensorflow]
task: Text Classification
language: en
nav_key: models
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

Given a proposal on a socially important issue, this model classifies it according to its topic.

## Predicted Entities

`Democracy`, `Digital`, `EU_In_The_World`, `Economy`, `Education`, `Green_Deal`, `Health`, `Migration`, `Other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_proposal_topic_en_1.0.0_3.0_1676594573703.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_proposal_topic_en_1.0.0_3.0_1676594573703.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentence_embeddings = nlp.UniversalSentenceEncoder.pretrained()\
    .setInputCols("document")\
    .setOutputCol("sentence_embeddings")

classifier = legal.ClassifierDLModel.pretrained("legclf_proposal_topic", "en", "legal/models")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("class")

clf_pipeline = nlp.Pipeline(stages=[
            document_assembler, 
            sentence_embeddings,
            classifier
            ])

empty_df = spark.createDataFrame([['']]).toDF("text")

model = clf_pipeline.fit(empty_df)

text = ["""In order to involve young people in the European Union, they need to understand the role, importance, and impact of the European Union on their lives and how they can contribute to the EU. I believe that many Europeans do not know the values of Europe, how they can contribute to the EU, etc. To do this, it was necessary to create an education program on the European Union that could cut across all countries, including a discipline on the EU, visits by young people to the European institutions, and a 'channel of communication' between young people and the EU. The same could be done for older people in senior universities."""]

data = spark.createDataFrame([text]).toDF("text")

result = model.transform(data)
```

</div>

## Results

```bash
+-----------+
|     result|
+-----------+
|[Education]|
+-----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_proposal_topic|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|22.6 MB|

## References

Training dataset available [here](https://touche.webis.de/clef23/touche23-web/multilingual-stance-classification.html#data)

## Benchmarking

```bash
label            precision  recall  f1-score  support 
Democracy        0.86       0.90    0.88      62      
Digital          0.85       0.80    0.82      35      
EU_In_The_World  0.78       0.72    0.75      39      
Economy          0.82       0.77    0.80      43      
Education        0.89       0.87    0.88      46      
Green_Deal       0.85       0.92    0.88      49      
Health           0.87       0.95    0.91      21      
Migration        0.86       0.89    0.87      27      
Other            1.00       0.97    0.98      32      
accuracy         -          -       0.86      354     
macro-avg        0.86       0.87    0.86      354     
weighted-avg     0.86       0.86    0.86      354 
```
