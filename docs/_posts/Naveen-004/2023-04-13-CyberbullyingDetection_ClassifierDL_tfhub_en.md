---
layout: model
title: Cyberbullying Detection
author: Naveen-004
name: CyberbullyingDetection_ClassifierDL_tfhub
date: 2023-04-13
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.4.0
spark_version: 3.0
supported: false
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Identify cyberbullying using a multi-class classification framework that distinguishes six different types of cyberbullying. We have used a Twitter dataset from Kaggle and applied various techniques such as text cleaning, data augmentation, document assembling, universal sentence encoding and tensorflow classification model to process and analyze the data. We have also used snscrape to retrieve tweet data for validating our model’s performance. Our results show that our model achieved an accuracy of 85% for testing data and 89% for training data.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/drive/1xaIlDtpiGzf14EA1umhJoOXI0FZaYtRc?authuser=4#scrollTo=os2C1v2WW1Hi){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/Naveen-004/CyberbullyingDetection_ClassifierDL_tfhub_en_4.4.0_3.0_1681363209630.zip){:.button.button-orange}
[Copy S3 URI](s3://community.johnsnowlabs.com/Naveen-004/CyberbullyingDetection_ClassifierDL_tfhub_en_4.4.0_3.0_1681363209630.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
    .setInputCol("cleaned_text")\
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained(name="tfhub_use_lg", lang="en")\
 .setInputCols("document")\
 .setOutputCol("sentence_embeddings")\
 .setDimension(768)

classifierdl = ClassifierDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("cyberbullying_type")\
  .setBatchSize(16)\
  .setMaxEpochs(42)\
  .setDropout(0.4) \
  .setEnableOutputLogs(True)\
  .setLr(4e-3)
use_clf_pipeline = Pipeline(
    stages = [documentAssembler,
        use,
        classifierdl])
```

</div>

## Results

```bash
           precision    recall  f1-score   support

                age       0.94      0.96      0.95       796
          ethnicity       0.94      0.94      0.94       810
             gender       0.87      0.86      0.86       816
  not_cyberbullying       0.74      0.67      0.70       766
other_cyberbullying       0.67      0.71      0.69       775
           religion       0.94      0.96      0.95       731

           accuracy                           0.85      4694
          macro avg       0.85      0.85      0.85      4694
       weighted avg       0.85      0.85      0.85      4694

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|CyberbullyingDetection_ClassifierDL_tfhub|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.0+|
|License:|Open Source|
|Edition:|Community|
|Language:|en|
|Size:|811.9 MB|

## Included Models

- DocumentAssembler
- UniversalSentenceEncoder
- ClassifierDLModel