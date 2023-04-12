---
layout: model
title: Cyberbullying detection
author: Naveen-004
name: Cyberbullying_Detection_pipeline
date: 2023-04-12
tags: [en, open_source]
task: Text Classification
language: en
edition: Spark NLP 4.3.0
spark_version: 3.0
supported: false
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Our research project aims to tackle the issue of cyberbullying through the use of SparkNLP, a powerful and scalable natural language processing library. 

We developed a cyberbullying detection model by training it on a dataset of approx. 48,000 tweets taken  from Kaggle. 

We performed feature extraction of text data using the Universal Sentence encoder (USE) and built the classification model using ClassifierDL which uses Deep neural networks, which utilizes USE as an input for text classification. 

To address the problem of class imbalance, we implemented text augmentation.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/Naveen-004/Cyberbullying_Detection_pipeline_en_4.3.0_3.0_1681288493511.zip){:.button.button-orange}
[Copy S3 URI](s3://community.johnsnowlabs.com/Naveen-004/Cyberbullying_Detection_pipeline_en_4.3.0_3.0_1681288493511.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|Cyberbullying_Detection_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.3.0+|
|License:|Open Source|
|Edition:|Community|
|Language:|en|
|Size:|811.9 MB|

## Included Models

- DocumentAssembler
- UniversalSentenceEncoder
- ClassifierDLModel