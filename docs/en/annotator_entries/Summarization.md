{%- capture title -%}
Summarization
{%- endcapture -%}

{%- capture model_description -%}
Fitted model produced by `Summarization`. It orchestrates the resolved summarization delegate at
the DataFrame level: prompt building, long document chunking, delegate inference, output cleanup,
and transparency metadata.

The summarization `method` is bound at fit time. A `SummarizationModel` only holds the delegate
of the method it was fitted with (`llm`, `encoder_decoder`, or `extractive`). Task parameters
that shape the output (summary length, style, focus, long document strategy, MMR/position
weights, generation settings) can be changed freely on the fitted model. Saving the model
persists the underlying delegate weights, so fitted pipelines reload without network access.

This is the instantiated model of the `Summarization` estimator. To build one, fit a
`Summarization` stage (see the documentation of that class).

**Note:** As a DataFrame level orchestrator, this annotator is not supported in `LightPipeline`.

**Note:** The `encoder_decoder` method pins inference to a single Spark partition (the BART
backend is not thread-safe); the `llm` and `extractive` methods keep the input partitioning. A
single fitted model instance is not safe for concurrent `transform` calls, since `transform`
configures the shared delegate in place — use one instance per concurrent caller.

**Note:** With the `llm` method the document text is embedded into the prompt, so a document
containing instructions can influence its own summary (prompt injection). The built-in system
prompt reduces but does not eliminate this.

**Note:** `transform` caches the input row ids and the computed summaries to keep the returned
DataFrame consistent. These caches cannot be released through this API; they are freed when the
SparkSession ends or via `spark.catalog.clearCache()` (which clears all cached data).
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture model_python_example -%}
>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> documentAssembler = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> summarizer = Summarization() \
...     .setInputCols(["document"]) \
...     .setOutputCol("summary") \
...     .setMethod("extractive") \
...     .setMaxSummaryLength(60)
>>> pipeline = Pipeline().setStages([documentAssembler, summarizer])
>>> data = spark.createDataFrame([["<a long document ...>"]]).toDF("text")
>>> model = pipeline.fit(data)
>>> # the fitted stage is a SummarizationModel; its parameters can still be tuned
>>> result = model.transform(data)
>>> result.select("summary.result").show(truncate=False)
{%- endcapture -%}

{%- capture model_scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val summarizer = new Summarization()
  .setInputCols("document")
  .setOutputCol("summary")
  .setMethod("extractive")
  .setMaxSummaryLength(60)

val pipeline = new Pipeline().setStages(Array(documentAssembler, summarizer))

val data = Seq("<a long document ...>").toDF("text")
val model = pipeline.fit(data)
val result = model.transform(data)
result.select("summary.result").show(false)
{%- endcapture -%}

{%- capture model_api_link -%}
[SummarizationModel](/api/com/johnsnowlabs/nlp/annotators/seq2seq/SummarizationModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[SummarizationModel](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/summarization/index.html#sparknlp.annotator.seq2seq.summarization.SummarizationModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[SummarizationModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/SummarizationModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
High level, task oriented document summarization. `Summarization` is a zero configuration
estimator: state *what* you want (summary length, style, focus) and the annotator decides *how*
to produce it (model selection, prompting, generation settings, and long document handling). No
prompt writing or model choice is required, because `fit()` resolves and downloads a sensible default
model for the chosen method.

Three summarization methods are supported, each with an automatically selected default model:

* `llm` (default): an instruction tuned GGUF LLM run with llama.cpp (via `AutoGGUFModel`). The
  annotator owns the summarization prompt, the system prompt, safe generation defaults, and
  reasoning mode suppression. Default model: `qwen3_4b_q8_0_gguf`.
* `encoder_decoder`: a specialized abstractive summarization model (`BartTransformer`, DistilBART
  fine tuned on XSum). Default model: `distilbart_xsum_12_6`.
* `extractive`: selects the most central sentences from the original document using sentence
  embeddings, position augmented centrality, and MMR redundancy control (a PacSum style ranker).
  The output contains only text taken from the source. Default model: `all_mpnet_base_v2`.

Documents longer than the model context are handled automatically
(`setLongDocumentStrategy("auto" | "truncate" | "hierarchical")`): the document is split at
sentence boundaries into overlapping chunks, each chunk is summarized, and the intermediate
summaries are combined and summarized again. `auto` and `hierarchical` currently behave
identically (a document that fits is summarized in a single pass either way); only `truncate`
differs, cutting the document to the context budget.

The public API exposes summarization concepts rather than model concepts:

```scala
summarizer
  .setMethod("llm")
  .setMaxSummaryLength(250)
  .setMinSummaryLength(50)
  .setSummaryStyle("concise")        // concise | detailed | bullets (llm)
  .setFocus("main findings")          // free text focus hint (llm)
  .setLongDocumentStrategy("auto")
```

Advanced parameters (`setModel`, `setNumBeams`, `setTemperature`, `setTopP`, `setChunkSize`,
`setChunkOverlap`, `setMmrLambda`, `setPositionBias`, `setGpuLayers`) are optional; most users
never need them. Parameters that do not apply to the selected method are ignored with a logged
warning. On CPU only clusters, set `setGpuLayers(0)` for the `llm` method.

Regardless of method, the output is a single `DOCUMENT` annotation whose metadata reports the
selected `method`, `model`, `engine`, estimated original/summary token counts, number of chunks,
and the long document strategy used.

For extended examples of usage, see the
[example notebook](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/text/english/text-summarization/Summarization_Annotator.ipynb).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture approach_python_example -%}
>>> import sparknlp
>>> from sparknlp.base import *
>>> from sparknlp.annotator import *
>>> from pyspark.ml import Pipeline
>>> documentAssembler = DocumentAssembler() \
...     .setInputCol("text") \
...     .setOutputCol("document")
>>> # zero configuration: the default method (llm) and its default model are resolved at fit()
>>> summarizer = Summarization() \
...     .setInputCols(["document"]) \
...     .setOutputCol("summary") \
...     .setMethod("encoder_decoder") \
...     .setMaxSummaryLength(80)
>>> pipeline = Pipeline().setStages([documentAssembler, summarizer])
>>> data = spark.createDataFrame([[
...     "Spark NLP is an open source text processing library for Python, Java and Scala. "
...     "It provides production grade, scalable, and trainable versions of the latest research "
...     "in natural language processing."]]).toDF("text")
>>> result = pipeline.fit(data).transform(data)
>>> result.select("summary.result").show(truncate=False)
>>> result.select("summary.metadata").show(truncate=False)
{%- endcapture -%}

{%- capture approach_scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import spark.implicits._

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val summarizer = new Summarization()
  .setInputCols("document")
  .setOutputCol("summary")
  .setMethod("encoder_decoder")
  .setMaxSummaryLength(80)

val pipeline = new Pipeline().setStages(Array(documentAssembler, summarizer))

val data = Seq(
  "Spark NLP is an open source text processing library for Python, Java and Scala. " +
    "It provides production grade, scalable, and trainable versions of the latest research " +
    "in natural language processing.").toDF("text")

val result = pipeline.fit(data).transform(data)
result.select("summary.result").show(false)
result.select("summary.metadata").show(false)
{%- endcapture -%}

{%- capture approach_api_link -%}
[Summarization](/api/com/johnsnowlabs/nlp/annotators/seq2seq/Summarization)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[Summarization](/api/python/reference/autosummary/sparknlp/annotator/seq2seq/summarization/index.html#sparknlp.annotator.seq2seq.summarization.Summarization)
{%- endcapture -%}

{%- capture approach_source_link -%}
[Summarization](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/seq2seq/Summarization.scala)
{%- endcapture -%}

{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_api_link=model_api_link
model_python_api_link=model_python_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
approach_python_api_link=approach_python_api_link
approach_source_link=approach_source_link
%}
