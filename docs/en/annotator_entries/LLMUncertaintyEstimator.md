{%- capture title -%}
LLMUncertaintyEstimator
{%- endcapture -%}

{%- capture description -%}
Estimates how uncertain an LLM is about a completion it generated, from one or more sampled
completions.

This annotator computes no logits and loads no model of its own: it is a pure post-processor
over annotations produced upstream in the pipeline. Two families of methods are supported (see
[Bakman et al. 2025](https://arxiv.org/abs/2506.01114) for why these specific methods were
chosen: they are the only ones the paper found to keep low error across calibration-set
distribution shift):

- **Black box** (`semanticEntropy`, `eccentricity`): needs multiple sampled completions for the
  same prompt (`AutoGGUFModel.setNumSamples(n)`) plus a way to tell which samples mean the same
  thing - either the default `similarityBackend="embeddings"` (cosine similarity of an
  additional sentence-embeddings input column, e.g. from `MPNetEmbeddings` or `MiniLMEmbeddings`
  - both Sentence-BERT models trained on NLI/STS data for this exact "are these two answers
  equivalent" task, unlike retrieval-oriented embedders such as E5 or BGE - alongside the
  completions column), or `similarityBackend="nli"` (bidirectional entailment,
  needs a `SampleEntailmentMatrix` stage run over the same completions column beforehand
  instead of an embeddings column).
- **White box** (`mars`, `meanLogProb`, `perplexity`, `predictiveEntropy`): needs per-token log
  probabilities (`AutoGGUFModel.setOutputLogProbs(true)`, and for `predictiveEntropy` also
  `setNProbs(k > 1)`). `mars` additionally needs a `MarsTokenImportance` stage run over the same
  completions column beforehand. Unlike the black-box methods, these work with a single sample
  and cost about as much as one generation, since no resampling is needed.

`uncertainty_score` is oriented so that higher means more uncertain; `confidence_score` is
`1 - uncertainty_score`. A raw uncertainty score on its own does not tell you whether an answer
should be trusted: the paper this annotator is based on found that decision thresholds must be
calibrated on data resembling your deployment distribution, or error rates rise sharply. Set the
`threshold` param (once calibrated on your own data) to get a boolean `is_reliable` metadata
flag; without it, only the raw score is emitted.

For extended examples of usage, see
[LLMUncertaintyEstimatorTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/uncertainty/LLMUncertaintyEstimatorTest.scala).
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

document = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

llm = AutoGGUFModel.pretrained() \
    .setInputCols(["document"]).setOutputCol("completions") \
    .setNumSamples(5).setTemperature(0.7)

embeddings = MPNetEmbeddings.pretrained() \
    .setInputCols(["completions"]).setOutputCol("sample_embeddings")

uncertainty = LLMUncertaintyEstimator() \
    .setInputCols(["completions", "sample_embeddings"]).setOutputCol("uncertainty") \
    .setMethods(["semanticEntropy"])

pipeline = Pipeline().setStages([document, llm, embeddings, uncertainty])

data = spark.createDataFrame([["What is the capital of France?"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("uncertainty.metadata").show(truncate=False)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")

val llm = AutoGGUFModel.pretrained()
  .setInputCols("document").setOutputCol("completions")
  .setNumSamples(5).setTemperature(0.7f)

val embeddings = MPNetEmbeddings.pretrained()
  .setInputCols("completions").setOutputCol("sample_embeddings")

val uncertainty = new LLMUncertaintyEstimator()
  .setInputCols("completions", "sample_embeddings").setOutputCol("uncertainty")
  .setMethods(Array("semanticEntropy"))

val pipeline = new Pipeline().setStages(Array(document, llm, embeddings, uncertainty))

val data = Seq("What is the capital of France?").toDF("text")
val result = pipeline.fit(data).transform(data)
result.select("uncertainty.metadata").show(truncate = false)
{%- endcapture -%}

{%- capture api_link -%}
[LLMUncertaintyEstimator](/api/com/johnsnowlabs/nlp/annotators/uncertainty/LLMUncertaintyEstimator)
{%- endcapture -%}

{%- capture python_api_link -%}
[LLMUncertaintyEstimator](/api/python/reference/autosummary/sparknlp/annotator/uncertainty/llm_uncertainty_estimator/index.html)
{%- endcapture -%}

{%- capture source_link -%}
[LLMUncertaintyEstimator](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/uncertainty/LLMUncertaintyEstimator.scala)
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
