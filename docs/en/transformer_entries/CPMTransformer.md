{%- capture title -%}
CPMTransformer
{%- endcapture -%}

{%- capture description -%}
The CPM model was proposed in [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://huggingface.co/papers/2012.00413) by Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.

This model was contributed by [canwenxu](https://huggingface.co/canwenxu). The original implementation can be found here: [https://github.com/TsinghuaAI/CPM-Generate](TsinghuaAI/CPM-Generate on GitHub)

Pretrained models can be loaded with `pretrained` of the companion object:
```scala
val seq2seq = CPMTransformer.pretrained("mini_cpm_2b_8bit","xx") 
    .setInputCols(Array("documents")) 
    .setOutputCol("generation")
```
The default model is `"mini_cpm_2b_8bit"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models?annotator=CPMTransformer).

Spark NLP also supports Hugging Face transformer-based code generation models. Learn more here:  
- [Import models into Spark NLP](https://github.com/JohnSnowLabs/spark-nlp/discussions/5669)

**Sources** :

- [CPM: A Large-scale Generative Chinese Pre-trained Language Model](https://huggingface.co/papers/2012.00413)
- [CPM-Generate on GitHub](https://github.com/TsinghuaAI/CPM-Generate)

**Paper abstract**

*Pre-trained Language Models (PLMs) have proven to be beneficial for various downstream NLP tasks. Recently, GPT-3, with 175 billion parameters and 570GB training data, drew a lot of attention due to the capacity of few-shot (even zero-shot) learning. However, applying GPT-3 to address Chinese NLP tasks is still challenging, as the training corpus of GPT-3 is primarily English, and the parameters are not publicly available. In this technical report, we release the Chinese Pre-trained Language Model (CPM) with generative pre-training on large-scale Chinese training data. To the best of our knowledge, CPM, with 2.6 billion parameters and 100GB Chinese training data, is the largest Chinese pre-trained language model, which could facilitate several downstream Chinese NLP tasks, such as conversation, essay generation, cloze test, and language understanding. Extensive experiments demonstrate that CPM achieves strong performance on many NLP tasks in the settings of few-shot (even zero-shot) learning.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
GENERATION
{%- endcapture -%}

{%- capture api_link -%}
[CPMTransformer](/api/com/johnsnowlabs/nlp/annotators/seq2seq/CPMTransformer.html)
{%- endcapture -%}

{%- capture python_api_link -%}
[CPMTransformer](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/cpm_transformer/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[CPMTransformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/CPMTransformer.scala)
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import CPMTransformer
from pyspark.ml import Pipeline

test_data = spark.createDataFrame([
    (1, "Leonardo Da Vinci invented the microscope?")
], ["id", "text"])

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("documents")

mini_cpm = CPMTransformer.pretrained() \
    .setInputCols(["documents"]) \
    .setOutputCol("generation") \
    .setMaxOutputLength(50) \
    .setDoSample(True)

pipeline = Pipeline().setStages([
    document_assembler,
    mini_cpm
])

model = pipeline.fit(test_data)
results = model.transform(test_data)

results.select("generation.result").show(truncate=False)

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                           |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[Leonardo Da Vinci invented the microscope?\n\n  - Leonardo da Vinci, a renowned Italian polymath, is often credited with inventing the microscope. However, this claim is not accurate.\n\n  - The microscope was actually invented by a Dutch scientist named Antonie van Leeu]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.generative.CPMTransformer
import org.apache.spark.ml.Pipeline

val testData = Seq(
  (1, "Leonardo Da Vinci invented the microscope?")
).toDF("id", "text")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("documents")

val miniCpm = CPMTransformer.pretrained()
  .setInputCols("documents")
  .setOutputCol("generation")
  .setMaxOutputLength(50)
  .setDoSample(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  miniCpm
))

val model = pipeline.fit(testData)
val results = model.transform(testData)

results.select("generation.result").show(false)

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                                                                                                                                           |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|[Leonardo Da Vinci invented the microscope?\n\n  - Leonardo da Vinci, a renowned Italian polymath, is often credited with inventing the microscope. However, this claim is not accurate.\n\n  - The microscope was actually invented by a Dutch scientist named Antonie van Leeu]|
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}