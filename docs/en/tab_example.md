---
layout: docs
header: true
title: Enterprise Spark NLP
permalink: /docs/en/tab_example
key: docs-licensed-install
modify_date: "2021-03-09"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

<div class="tabs-box tabs-new" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

<div class="highlighter-rouge language-python" markdown="1">
{% include programmingLanguageSelectPythons.html %}

<div class="python-inner python-spark-nlp-jsl" markdown="1">

 ```python
...
pos = PerceptronModel.pretrained("pos_clinical","en","clinical/models")\
	.setInputCols(["token","sentence"])\
	.setOutputCol("pos")

pos_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(pos_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))
result = light_pipeline.fullAnnotate("""He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.""")
```

</div>
<div class="python-inner python-johnsnowlabs" markdown="1">

```python
...
pos = PerceptronModel.pretrained("pos_clinical","en","clinical/models")\
    .setInputCols(["token","sentence"])\
    .setOutputCol("pos")

pos_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(pos_pipeline.fit(spark.createDataFrame([[""]]).toDF("text")))
result = light_pipeline.fullAnnotate("""He was given boluses of MS04 with some effect, he has since been placed on a PCA - he take 80mg of oxycontin at home, his PCA dose is ~ 2 the morphine dose of the oxycontin, he has also received ativan for anxiety.""")
```

</div>
</div>

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
