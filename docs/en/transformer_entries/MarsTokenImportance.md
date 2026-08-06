{%- capture title -%}
MarsTokenImportance
{%- endcapture -%}

{%- capture description -%}
Computes MARS per-token importance weights for sampled LLM answers, given the question they
answer, using a BERT token-classification model
([duygunuryldz/MARS](https://huggingface.co/duygunuryldz/MARS) by default -
[Bakman et al. 2024](https://arxiv.org/abs/2402.11756)).

This is a plumbing annotator for `LLMUncertaintyEstimator`'s `mars` method: it does not itself
produce an uncertainty score, it only attaches a `token_importance` metadata field (a JSON
array of `{"begin", "end", "importance"}` character-offset spans into the answer) that
`LLMUncertaintyEstimator` reads and combines with the answer's per-token log probabilities
(from `AutoGGUFModel.setOutputLogProbs(true)`).

Takes two DOCUMENT input columns, in this order: the question, and the sampled answer(s) to
score (one row may carry several sampled answers, e.g. from
`AutoGGUFModel.setNumSamples(n)`; every sample in a row is scored against that row's single
question).

Pretrained models can be loaded with `pretrained` of the companion object, or a local ONNX
export loaded with `loadSavedModel`:

```scala
val marsImportance = MarsTokenImportance.loadSavedModel("path/to/mars_onnx", spark)
  .setInputCols("question", "completions")
  .setOutputCol("token_importance")
```
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

question = DocumentAssembler().setInputCol("question").setOutputCol("question_doc")
answer = DocumentAssembler().setInputCol("answer").setOutputCol("answer_doc")

mars = MarsTokenImportance.pretrained() \
    .setInputCols(["question_doc", "answer_doc"]).setOutputCol("token_importance")

pipeline = Pipeline().setStages([question, answer, mars])

data = spark.createDataFrame([
    ("What is the capital of France?", "Paris is the capital of France.")
]).toDF("question", "answer")
result = pipeline.fit(data).transform(data)
result.select("token_importance.metadata").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val question = new DocumentAssembler().setInputCol("question").setOutputCol("question_doc")
val answer = new DocumentAssembler().setInputCol("answer").setOutputCol("answer_doc")

val mars = MarsTokenImportance.pretrained()
  .setInputCols("question_doc", "answer_doc").setOutputCol("token_importance")

val pipeline = new Pipeline().setStages(Array(question, answer, mars))

val data = Seq(
  ("What is the capital of France?", "Paris is the capital of France.")
).toDF("question", "answer")
val result = pipeline.fit(data).transform(data)
result.select("token_importance.metadata").show(truncate = false)
{%- endcapture -%}

{%- capture api_link -%}
[MarsTokenImportance](/api/com/johnsnowlabs/nlp/annotators/uncertainty/MarsTokenImportance)
{%- endcapture -%}

{%- capture python_api_link -%}
[MarsTokenImportance](/api/python/reference/autosummary/sparknlp/annotator/uncertainty/mars_token_importance/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[MarsTokenImportance](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/uncertainty/MarsTokenImportance.scala)
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
