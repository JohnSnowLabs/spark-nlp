````markdown
{%- capture title -%}
AutoGGUFReranker
{%- endcapture -%}

{%- capture description -%}
Annotator that uses the llama.cpp library to rerank text documents based on their relevance to
a given query using GGUF-format reranking models.

This annotator is specifically designed for text reranking tasks, where multiple documents or
text passages are ranked according to their relevance to a query. It uses specialized
reranking models in GGUF format that output relevance scores for each input document.

The reranker takes a query (set via `setQuery`) and a list of documents, then returns the same
documents with added metadata containing relevance scores. The documents are processed in
batches and each receives a `relevance_score` in its metadata indicating how relevant it is to
the provided query.

For settable parameters, and their explanations, see [HasLlamaCppInferenceProperties](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/HasLlamaCppInferenceProperties.scala), [HasLlamaCppModelProperties](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/HasLlamaCppModelProperties.scala) and refer to
the llama.cpp documentation of
[server.cpp](https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server)
for more information.

If the parameters are not set, the annotator will default to use the parameters provided by
the model.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val reranker = AutoGGUFReranker.pretrained()
  .setInputCols("document")
  .setOutputCol("reranked_documents")
  .setQuery("A man is eating pasta.")
```

The default model is `"bge-reranker-v2-m3-Q4_K_M"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://sparknlp.org/models).

For extended examples of usage, see the
[AutoGGUFRerankerTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFRerankerTest.scala)
and the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFReranker.ipynb).

**Note**: This annotator is designed for reranking tasks and requires setting a query using `setQuery`.
The query represents the search intent against which documents will be ranked. Each input
document receives a relevance score in the output metadata.

To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
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
>>> reranker = AutoGGUFReranker.pretrained() \
...     .setInputCols(["document"]) \
...     .setOutputCol("reranked_documents") \
...     .setBatchSize(4) \
...     .setQuery("A man is eating pasta.") \
...     .setNGpuLayers(99)
>>> pipeline = Pipeline().setStages([document, reranker])
>>> data = spark.createDataFrame([
...     ["A man is eating food."],
...     ["A man is eating a piece of bread."],
...     ["The girl is carrying a baby."],
...     ["A man is riding a horse."]
... ]).toDF("text")
>>> result = pipeline.fit(data).transform(data)
>>> result.select("reranked_documents").show(truncate = False)
+-------------------------------------------------------------------------------------------+
|reranked_documents                                                                         |
+-------------------------------------------------------------------------------------------+
|[{document, 0, 20, A man is eating food., {query -> A man is eating pasta., relevance_...}]|
|[{document, 0, 31, A man is eating a piece of bread., {query -> A man is eating pasta.,...}]|
|[{document, 0, 27, The girl is carrying a baby., {query -> A man is eating pasta., rel...}]|
|[{document, 0, 22, A man is riding a horse., {query -> A man is eating pasta., relevan...}]|
+-------------------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val reranker = AutoGGUFReranker
  .pretrained("bge-reranker-v2-m3-Q4_K_M")
  .setInputCols("document")
  .setOutputCol("reranked_documents")
  .setBatchSize(4)
  .setQuery("A man is eating pasta.")
  .setNGpuLayers(99)

val pipeline = new Pipeline().setStages(Array(document, reranker))

val data = Seq(
  "A man is eating food.",
  "A man is eating a piece of bread.",
  "The girl is carrying a baby.",
  "A man is riding a horse."
).toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("reranked_documents").show(truncate = false)
+-------------------------------------------------------------------------------------------+
|reranked_documents                                                                         |
+-------------------------------------------------------------------------------------------+
|[{document, 0, 20, A man is eating food., {query -> A man is eating pasta., relevance_...}]|
|[{document, 0, 31, A man is eating a piece of bread., {query -> A man is eating pasta.,...}]|
|[{document, 0, 27, The girl is carrying a baby., {query -> A man is eating pasta., rel...}]|
|[{document, 0, 22, A man is riding a horse., {query -> A man is eating pasta., relevan...}]|
+-------------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[AutoGGUFReranker](/api/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFReranker)
{%- endcapture -%}

{%- capture python_api_link -%}
[AutoGGUFReranker](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/auto_gguf_reranker/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[AutoGGUFReranker](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFReranker.scala)
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
````
