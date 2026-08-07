{%- capture title -%}
SampleEntailmentMatrix
{%- endcapture -%}

{%- capture description -%}
Computes a bidirectional-entailment matrix over a row's sampled LLM answers, using a BERT
sequence-classification model trained on NLI.

This is the faithful-to-the-literature alternative to `LLMUncertaintyEstimator`'s default
`similarityBackend="embeddings"`: [Kuhn et al. 2023](https://arxiv.org/abs/2302.09664)'s
Semantic Entropy clusters samples by checking whether each pair of samples entails the other (in
both directions), rather than by embedding similarity.

This is a plumbing annotator, like `MarsTokenImportance`: it does not itself produce an
uncertainty score, it only attaches an `entailment_matrix` metadata field (a row-major N x N
JSON array of entailment probabilities) that `LLMUncertaintyEstimator` reads when
`setSimilarityBackend("nli")` is set.

Scoring all ordered pairs of N samples needs `N*(N-1)` model calls - this grows fast.
`maxSamplesForNli` (default `10`, so up to 90 calls per row) guards against silently issuing
very large batches.

**No pretrained model is published yet.** `pretrained()` has no working hub model - Spark NLP's
`.pretrained()` deserializes a model in the format the *calling class itself* wrote it in (here,
an ONNX file under `sample_entailment_matrix_onnx`), and no BERT NLI checkpoint has ever been
published to the hub in that exact format. Use `loadSavedModel` with a self-exported model
instead: export [textattack/bert-base-uncased-MNLI](https://huggingface.co/textattack/bert-base-uncased-MNLI)
to ONNX (`torch.onnx.export`, `dynamo=False`), lay it out as `<model_dir>/model.onnx` plus
`<model_dir>/assets/vocab.txt` and `<model_dir>/assets/labels.txt`, and load with
`SampleEntailmentMatrix.loadSavedModel("<model_dir>", spark)`. This checkpoint's label order is
`contradiction, entailment, neutral` (confirmed empirically, not the textbook GLUE MNLI order -
its `config.json` has no `id2label` at all).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document = DocumentAssembler().setInputCol("text").setOutputCol("document")

entailment = SampleEntailmentMatrix.loadSavedModel("<model_dir>", spark) \
    .setInputCols(["document"]).setOutputCol("entailment")

pipeline = Pipeline().setStages([document, entailment])

data = spark.createDataFrame([["Paris."], ["The capital is Paris."], ["London."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("entailment.metadata").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val entailment = SampleEntailmentMatrix.loadSavedModel("<model_dir>", spark)
  .setInputCols("document").setOutputCol("entailment")

val pipeline = new Pipeline().setStages(Array(document, entailment))

val data = Seq("Paris.", "The capital is Paris.", "London.").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("entailment.metadata").show(truncate = false)
{%- endcapture -%}

{%- capture api_link -%}
[SampleEntailmentMatrix](/api/com/johnsnowlabs/nlp/annotators/uncertainty/SampleEntailmentMatrix)
{%- endcapture -%}

{%- capture python_api_link -%}
[SampleEntailmentMatrix](/api/python/reference/autosummary/sparknlp/annotator/uncertainty/sample_entailment_matrix/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[SampleEntailmentMatrix](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/uncertainty/SampleEntailmentMatrix.scala)
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
