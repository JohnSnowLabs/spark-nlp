---
layout: model
title: RelationExtractionModel Clinical
author: John Snow Labs
name: re_clinical_en
date: 2020-08-09
tags: [re, en, re_clinical]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Models the set of clinical relations defined in the 2010 i2b2 relation challenge.

{:.h2_title}
## Included Relations 

TrIP: A certain treatment has improved or cured a medical problem (eg, ‘infection resolved with antibiotic course’)

TrWP: A patient's medical problem has deteriorated or worsened because of or in spite of a treatment being administered (eg, ‘the tumor was growing despite the drain’)

TrCP: A treatment caused a medical problem (eg, ‘penicillin causes a rash’)

TrAP: A treatment administered for a medical problem (eg, ‘Dexamphetamine for narcolepsy’)

TrNAP: The administration of a treatment was avoided because of a medical problem (eg, ‘Ralafen which is contra-indicated because of ulcers’)

TeRP: A test has revealed some medical problem (eg, ‘an echocardiogram revealed a pericardial effusion’)

TeCP: A test was performed to investigate a medical problem (eg, ‘chest x-ray done to rule out pneumonia’)

PIP: Two problems are related to each other (eg, ‘Azotemia presumed secondary to sepsis’)


{:.btn-box}
[Live Demo](){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_clinical_en_2.5.5_2.4_1596928426753.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, PerceptronModel, NerDLModel, NerConverter, DependencyParserModel, RelationExtractionModel.

The precision of the RE model is controlled by "setMaxSyntacticDistance(4)", which sets the maximum syntactic distance between named entities to 4. A larger value will improve recall at the expense at lower precision.


{% include programmingLanguageSelectScalaPython.html %}


```python


clinical_re_Model = RelationExtractionModel()\
    .pretrained("re_clinical", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\
    .setRelationPairs(["problem-test", "problem-treatment"]) # Possible relation pairs. Default is all relations.

loaded_pipeline = Pipeline(stages=[clinical_re_Model])

empty_data = spark.createDataFrame([[""]]).toDF("text")

loaded_model = loaded_pipeline.fit(empty_data)

loaded_lmodel = LightPipeline(loaded_model)

annotations = loaded_lmodel.fullAnnotate(text)

rel_df = get_relations_df (annotations)
```

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|re_clinical_en_2.5.5_2.4|
|Type:|re|
|Compatibility:|Spark NLP 2.5.5|
|Edition:|Healthcare|
|License:|Enterprise|
|Spark inputs:|[embeddings, pos_tags, ner_chunks, dependencies]|
|Spark outputs:|[relations]|
|Language:|[en]|
|Case sensitive:|false|

{:.h2_title}
## Dataset used for training
Trained on augmented 2010 i2b2 challenge data with 'clinical_embeddings'.
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

{:.h2_title}
## Results
The output is a dataframe with a Relation column and a Confidence column.
![image](\assets\images\re_clinical.png)
