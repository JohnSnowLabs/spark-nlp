{%- capture title -%}
AutoGGUFEmbeddings
{%- endcapture -%}

{%- capture description -%}
Annotator that uses the llama.cpp library to generate text embeddings with large language
models.

The type of embedding pooling can be set with the `setPoolingType` method. The default is
`"MEAN"`. The available options are `"NONE"`, `"MEAN"`, `"CLS"`, and `"LAST"`.

If the parameters are not set, the annotator will default to use the parameters provided by
the model.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val autoGGUFEmbeddings = AutoGGUFEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("embeddings")
```

The default model is `"nomic-embed-text-v1.5.Q8_0.gguf"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models).

For extended examples of usage, see the
[AutoGGUFEmbeddingsTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFEmbeddingsTest.scala)
and the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFEmbeddings.ipynb).

**Note**: To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
the number of GPU layers with the `setNGpuLayers` method.

When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
according to your hardware to avoid out-of-memory errors.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> document = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> autoGGUFEmbeddings = AutoGGUFEmbeddings.pretrained() \
...     .setInputCols(["document"]) \
...     .setOutputCol("completions") \
...     .setBatchSize(4) \
...     .setNGpuLayers(99) \
...     .setPoolingType("MEAN")
>>> pipeline = Pipeline().setStages([document, autoGGUFEmbeddings])
>>> data = spark.createDataFrame([["The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones."]]).toDF("text")
>>> result = pipeline.fit(data).transform(data)
>>> result.select("completions").show()
+--------------------------------------------------------------------------------+
|                                                                      embeddings|
+--------------------------------------------------------------------------------+
|[[-0.034486726, 0.07770534, -0.15982522, -0.017873349, 0.013914132, 0.0365736...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val autoGGUFEmbeddings = AutoGGUFEmbeddings
  .pretrained()
  .setInputCols("document")
  .setOutputCol("embeddings")
  .setBatchSize(4)
  .setPoolingType("MEAN")

val pipeline = new Pipeline().setStages(Array(document, autoGGUFEmbeddings))

val data = Seq(
  "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones.")
  .toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("embeddings.embeddings").show(1, truncate=80)
+--------------------------------------------------------------------------------+
|                                                                      embeddings|
+--------------------------------------------------------------------------------+
|[[-0.034486726, 0.07770534, -0.15982522, -0.017873349, 0.013914132, 0.0365736...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%}
[AutoGGUFEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/AutoGGUFEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[AutoGGUFEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/auto_gguf_embeddings/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[AutoGGUFEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/AutoGGUFEmbeddings.scala)
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