---
layout: model
title: English image_classifier_vit_rust_image_classification_11 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: image_classifier_vit_rust_image_classification_11
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_11` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_rust_image_classification_11_en_4.1.0_3.0_1660171448218.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_rust_image_classification_11", "en")\
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    image_assembler,
    imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val imageAssembler = new ImageAssembler()\
.setInputCol("image")\
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification\
.pretrained("image_classifier_vit_rust_image_classification_11", "en")\
.setInputCols("image_assembler")\
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_rust_image_classification_11|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|---
layout: model
title: English pipeline_image_classifier_vit_rust_image_classification_12 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: pipeline_image_classifier_vit_rust_image_classification_12
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_12` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_rust_image_classification_12_en_4.2.1_3.0_1665570526567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_rust_image_classification_12', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_rust_image_classification_12", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_rust_image_classification_12|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification---
layout: model
title: English pipeline_image_classifier_vit_rust_image_classification_10 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: pipeline_image_classifier_vit_rust_image_classification_10
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_10` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_rust_image_classification_10_en_4.2.1_3.0_1665569934739.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_rust_image_classification_10', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_rust_image_classification_10", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_rust_image_classification_10|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification---
layout: model
title: English image_classifier_vit_rust_image_classification_11 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: image_classifier_vit_rust_image_classification_11
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_11` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_rust_image_classification_11_en_4.1.0_3.0_1660171448218.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_rust_image_classification_11", "en")\
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    image_assembler,
    imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val imageAssembler = new ImageAssembler()\
.setInputCol("image")\
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification\
.pretrained("image_classifier_vit_rust_image_classification_11", "en")\
.setInputCols("image_assembler")\
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_rust_image_classification_11|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|---
layout: model
title: English pipeline_image_classifier_vit_rust_image_classification_12 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: pipeline_image_classifier_vit_rust_image_classification_12
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_12` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_rust_image_classification_12_en_4.2.1_3.0_1665570526567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_rust_image_classification_12', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_rust_image_classification_12", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_rust_image_classification_12|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification---
layout: model
title: English pipeline_image_classifier_vit_rust_image_classification_10 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: pipeline_image_classifier_vit_rust_image_classification_10
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_10` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_rust_image_classification_10_en_4.2.1_3.0_1665569934739.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_rust_image_classification_10', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_rust_image_classification_10", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_rust_image_classification_10|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification---
layout: model
title: English pipeline_image_classifier_vit_rust_image_classification_1 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: pipeline_image_classifier_vit_rust_image_classification_1
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_1` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_rust_image_classification_1_en_4.2.1_3.0_1665534598923.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_rust_image_classification_1', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_rust_image_classification_1", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_rust_image_classification_1|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification---
layout: model
title: English image_classifier_vit_rust_image_classification_12 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: image_classifier_vit_rust_image_classification_12
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_12` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_rust_image_classification_12_en_4.1.0_3.0_1660171578567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_rust_image_classification_12", "en")\
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    image_assembler,
    imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val imageAssembler = new ImageAssembler()\
.setInputCol("image")\
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification\
.pretrained("image_classifier_vit_rust_image_classification_12", "en")\
.setInputCols("image_assembler")\
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_rust_image_classification_12|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|---
layout: model
title: English pipeline_image_classifier_vit_rust_image_classification_11 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: pipeline_image_classifier_vit_rust_image_classification_11
date: 2022-10-12
tags: [vit, en, images, open_source, pipeline]
task: Image Classification
language: en
edition: Spark NLP 4.2.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_11` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pipeline_image_classifier_vit_rust_image_classification_11_en_4.2.1_3.0_1665568778593.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

    pipeline = PretrainedPipeline('pipeline_image_classifier_vit_rust_image_classification_11', lang = 'en')
    annotations =  pipeline.transform(imageDF)
    
```
```scala

    val pipeline = new PretrainedPipeline("pipeline_image_classifier_vit_rust_image_classification_11", lang = "en")
    val annotations = pipeline.transform(imageDF)
    
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pipeline_image_classifier_vit_rust_image_classification_11|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.2.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|321.9 MB|

## Included Models

- ImageAssembler
- ViTForImageClassification---
layout: model
title: English image_classifier_vit_rust_image_classification_1 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: image_classifier_vit_rust_image_classification_1
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_1` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_rust_image_classification_1_en_4.1.0_3.0_1660166731268.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_rust_image_classification_1", "en")\
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    image_assembler,
    imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val imageAssembler = new ImageAssembler()\
.setInputCol("image")\
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification\
.pretrained("image_classifier_vit_rust_image_classification_1", "en")\
.setInputCols("image_assembler")\
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_rust_image_classification_1|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|---
layout: model
title: English image_classifier_vit_rust_image_classification_10 ViTForImageClassification from SummerChiam
author: John Snow Labs
name: image_classifier_vit_rust_image_classification_10
date: 2022-08-10
tags: [vit, en, images, open_source]
task: Image Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained VIT  model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`image_classifier_vit_rust_image_classification_10` is a English model originally trained by SummerChiam.


## Predicted Entities

`nonrust`, `rust`


## Predicted Entities

`nonrust`, `rust`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/image_classifier_vit_rust_image_classification_10_en_4.1.0_3.0_1660166162294.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

imageClassifier = ViTForImageClassification \
    .pretrained("image_classifier_vit_rust_image_classification_10", "en")\
    .setInputCols("image_assembler") \
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    image_assembler,
    imageClassifier,
])

pipelineModel = pipeline.fit(imageDF)

pipelineDF = pipelineModel.transform(imageDF)
```
```scala

val imageAssembler = new ImageAssembler()\
.setInputCol("image")\
.setOutputCol("image_assembler")

val imageClassifier = ViTForImageClassification\
.pretrained("image_classifier_vit_rust_image_classification_10", "en")\
.setInputCols("image_assembler")\
.setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))

val pipelineModel = pipeline.fit(imageDF)

val pipelineDF = pipelineModel.transform(imageDF)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|image_classifier_vit_rust_image_classification_10|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[image_assembler]|
|Output Labels:|[class]|
|Language:|en|
|Size:|321.9 MB|