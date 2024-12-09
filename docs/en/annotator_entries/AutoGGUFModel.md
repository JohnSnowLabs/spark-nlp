{%- capture title -%}
AutoGGUFModel
{%- endcapture -%}

{%- capture description -%}
Annotator that uses the llama.cpp library to generate text completions with large language
models.

For settable parameters, and their explanations, see [HasLlamaCppProperties](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/HasLlamaCppProperties.scala) and refer to
the llama.cpp documentation of
[server.cpp](https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server)
for more information.

If the parameters are not set, the annotator will default to use the parameters provided by
the model.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val autoGGUFModel = AutoGGUFModel.pretrained()
  .setInputCols("document")
  .setOutputCol("completions")
```

The default model is `"phi3.5_mini_4k_instruct_q4_gguf"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models).

For extended examples of usage, see the
[AutoGGUFModelTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFModelTest.scala)
and the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFModel.ipynb).

**Note**: To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
the number of GPU layers with the `setNGpuLayers` method.

When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
according to your hardware to avoid out-of-memory errors.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> document = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> autoGGUFModel = AutoGGUFModel.pretrained() \
...     .setInputCols(["document"]) \
...     .setOutputCol("completions") \
...     .setBatchSize(4) \
...     .setNPredict(20) \
...     .setNGpuLayers(99) \
...     .setTemperature(0.4) \
...     .setTopK(40) \
...     .setTopP(0.9) \
...     .setPenalizeNl(True)
>>> pipeline = Pipeline().setStages([document, autoGGUFModel])
>>> data = spark.createDataFrame([["Hello, I am a"]]).toDF("text")
>>> result = pipeline.fit(data).transform(data)
>>> result.select("completions").show(truncate = False)
+-----------------------------------------------------------------------------------------------------------------------------------+
|completions                                                                                                                        |
+-----------------------------------------------------------------------------------------------------------------------------------+
|[{document, 0, 78,  new user.  I am currently working on a project and I need to create a list of , {prompt -> Hello, I am a}, []}]|
+-----------------------------------------------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val autoGGUFModel = AutoGGUFModel
  .pretrained()
  .setInputCols("document")
  .setOutputCol("completions")
  .setBatchSize(4)
  .setNPredict(20)
  .setNGpuLayers(99)
  .setTemperature(0.4f)
  .setTopK(40)
  .setTopP(0.9f)
  .setPenalizeNl(true)

val pipeline = new Pipeline().setStages(Array(document, autoGGUFModel))

val data = Seq("Hello, I am a").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("completions").show(truncate = false)
+-----------------------------------------------------------------------------------------------------------------------------------+
|completions                                                                                                                        |
+-----------------------------------------------------------------------------------------------------------------------------------+
|[{document, 0, 78,  new user.  I am currently working on a project and I need to create a list of , {prompt -> Hello, I am a}, []}]|
+-----------------------------------------------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[AutoGGUFModel](/api/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFModel)
{%- endcapture -%}

{%- capture python_api_link -%}
[AutoGGUFModel](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/auto_gguf_model/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[AutoGGUFModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFModel.scala)
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