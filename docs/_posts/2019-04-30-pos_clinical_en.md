---
layout: model
title: POS Tagger Clinical
author: John Snow Labs
name: pos_clinical
class: PerceptronModel
language: en
repository: clinical/models
date: 2019-04-30
task: Part of Speech Tagging
edition: Healthcare NLP 2.0.2
spark_version: 2.4
tags: [clinical, licensed, pos,en]
supported: true
annotator: PerceptronModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Sets a Part-Of-Speech tag to each word within a sentence.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/pos_clinical_en_2.0.2_2.4_1556660550177.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_clinical","en","clinical/models")\
	.setInputCols(["token","sentence"])\
	.setOutputCol("pos")

pos_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(pos_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))
result = light_pipeline.fullAnnotate("""He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.""")
```

```scala
val pos = PerceptronModel.pretrained("pos_clinical","en","clinical/models")
	.setInputCols("token","sentence")
	.setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.").toDF("text")
val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.pos.clinical").predict("""He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.""")
```

</div>

{:.h2_title}
## Results

```bash
[Annotation(pos, 0, 1, NN, {'word': 'He'}),
Annotation(pos, 3, 5, VBD, {'word': 'was'}),
Annotation(pos, 7, 11, VVN, {'word': 'given'}),
Annotation(pos, 13, 19, NNS, {'word': 'boluses'}),
Annotation(pos, 21, 22, II, {'word': 'of'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|---------------------|
| Name:          | pos_clinical        |
| Type:   | PerceptronModel     |
| Compatibility: | Spark NLP 2.0.2+               |
| License:       | Licensed            |
| Edition:       | Official          |
|Input labels:        | [token, sentence]     |
|Output labels:       | [pos]                 |
| Language:      | en                  |
| Dependencies: | embeddings_clinical |

{:.h2_title}
## Data Source
Trained with MedPost dataset.