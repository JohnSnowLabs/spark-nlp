/*
 * Copyright 2017-2026 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.annotators.uncertainty

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.scalatest.flatspec.AnyFlatSpec

import java.io.File

/** End-to-end checks against the real NLI checkpoint.
  *
  * The batching in `NliClassification.entailmentProbabilities` pads every pair in a batch to the
  * batch's longest sequence and masks the padding out. That is only correct if the model actually
  * honours `attention_mask` - if it does not, or if the ONNX export pinned its sequence
  * dimension, batched scores drift from unbatched ones and `semanticEntropy` with
  * `similarityBackend="nli"` returns quietly wrong numbers instead of failing. Nothing about that
  * is observable without a real model, hence this test.
  */
class SampleEntailmentMatrixSlowTest extends AnyFlatSpec {

  private val modelPath = "models/bert_base_uncased_mnli_entailment_onnx"

  private lazy val model: SampleEntailmentMatrix = {
    ResourceHelper.spark
    SampleEntailmentMatrix.load(modelPath)
  }

  /** Deliberately mixed lengths, so a batch has real padding to mask out. */
  private val pairs: Seq[(String, String)] = Seq(
    ("Paris is the capital of France.", "The capital of France is Paris."),
    ("Paris is the capital of France.", "London is the capital of France."),
    (
      "A man is playing a guitar on stage in front of a very large and enthusiastic crowd.",
      "Someone is performing music."),
    ("The cat sat.", "A feline was seated."),
    (
      "It is raining heavily outside and the streets are already starting to flood badly.",
      "The weather is dry."),
    ("She bought three apples.", "She purchased fruit."),
    ("No.", "Yes."),
    (
      "The quick brown fox jumps over the lazy dog near the riverbank at sunset every evening.",
      "A fox jumps over a dog."))

  private def requireModel(): Unit =
    assume(new File(modelPath).exists(), s"Model directory not found: $modelPath")

  behavior of "NliClassification batching"

  it should "produce the same probabilities batched as one pair at a time" taggedAs SlowTest in {
    requireModel()
    val nli = model.getModelIfNotSet

    val unbatched = nli.entailmentProbabilities(pairs, 512, batchSize = 1)
    Seq(2, 3, 8, 16).foreach { batchSize =>
      val batched = nli.entailmentProbabilities(pairs, 512, batchSize)
      assert(batched.length == pairs.length)
      unbatched.zip(batched).zipWithIndex.foreach { case ((single, grouped), i) =>
        assert(
          math.abs(single - grouped) < 1e-4,
          s"batchSize=$batchSize pair $i: unbatched $single vs batched $grouped - padding is " +
            "changing the model's output, so attention_mask is not being honoured")
      }
    }
  }

  it should "be unaffected by where a pair sits within its batch" taggedAs SlowTest in {
    requireModel()
    val nli = model.getModelIfNotSet
    // The longest pair sets the padding width for the whole batch, so scoring a short pair
    // alongside it must match scoring it alongside another short one.
    val shortest = pairs.minBy { case (p, h) => p.length + h.length }
    val longest = pairs.maxBy { case (p, h) => p.length + h.length }

    val alone = nli.entailmentProbabilities(Seq(shortest), 512, batchSize = 1).head
    val withLong = nli.entailmentProbabilities(Seq(shortest, longest), 512, batchSize = 2).head
    val withLongFirst =
      nli.entailmentProbabilities(Seq(longest, shortest), 512, batchSize = 2)(1)

    assert(
      math.abs(alone - withLong) < 1e-4,
      s"$alone vs $withLong when padded up to a long pair")
    assert(
      math.abs(alone - withLongFirst) < 1e-4,
      s"$alone vs $withLongFirst at batch position 1")
  }

  it should "score a paraphrase above a contradiction" taggedAs SlowTest in {
    requireModel()
    val nli = model.getModelIfNotSet
    // Guards the labels.txt ordering: if the entailment label resolved to the wrong index, this
    // ranking inverts.
    val Array(paraphrase, contradiction) =
      nli.entailmentProbabilities(
        Seq(
          ("Paris is the capital of France.", "The capital of France is Paris."),
          ("Paris is the capital of France.", "London is the capital of France.")),
        512,
        batchSize = 2)

    assert(
      paraphrase > contradiction,
      s"paraphrase scored $paraphrase but contradiction scored $contradiction - check that " +
        "labels.txt lists the entailment class where the model actually puts it")
    assert(paraphrase > 0.5, s"a paraphrase should read as entailed, got $paraphrase")
  }

  it should "tell two answers apart when they differ only in a capitalised proper noun" taggedAs SlowTest in {
    requireModel()
    val nli = model.getModelIfNotSet
    // The distinction the NLI backend exists to make, in its hardest form: identical sentences
    // apart from the answer itself. An uncased checkpoint run with caseSensitive = true encodes
    // every one of these proper nouns as [UNK], so all three score identically and the backend
    // silently stops discriminating - which is what shipped in the first published artifact.
    val premise = "The capital of France is Paris."
    val Array(self, wrongCity, otherWrongCity) =
      nli.entailmentProbabilities(
        Seq(
          (premise, "The capital of France is Paris."),
          (premise, "The capital of France is London."),
          (premise, "The capital of France is Berlin.")),
        512,
        batchSize = 3)

    assert(
      self > wrongCity && self > otherWrongCity,
      s"a sentence must entail itself ($self) more than a contradicting one ($wrongCity, " +
        s"$otherWrongCity). Equal values across all three mean the proper nouns tokenized to " +
        "[UNK] - the model is uncased, so caseSensitive must be false.")
    assert(
      math.abs(wrongCity - otherWrongCity) > 1e-6 || wrongCity < 0.5,
      s"'London' and 'Berlin' scored identically ($wrongCity) - they are being encoded as the " +
        "same [UNK] token")
    assert(self > 0.9, s"a sentence should strongly entail itself, got $self")
  }

  it should "produce a symmetric-ish, 1.0-diagonal matrix through the annotator" taggedAs SlowTest in {
    requireModel()
    val spark = ResourceHelper.spark
    // Full sentences, not bare fragments. An MNLI model given "Paris." as a premise has almost
    // nothing to reason over and scores erratically (measured: it rates "Paris." as entailing
    // "London." at 0.96), so fragment-shaped completions are a poor fit for the NLI backend -
    // see the note in the annotator docs.
    val samples = Seq(
      "The capital of France is Paris.",
      "Paris is the capital city of France.",
      "The capital of France is London.")
    // DocumentAssembler only takes its multi-annotation-per-row path for an
    // ArrayType(StringType, containsNull = false) column, which `toDF` on a Seq does not produce.
    val schema = StructType(
      Seq(
        StructField(
          "completions_raw",
          ArrayType(StringType, containsNull = false),
          nullable = false)))
    val data = spark.createDataFrame(spark.sparkContext.parallelize(Seq(Row(samples))), schema)

    val documentAssembler = new com.johnsnowlabs.nlp.DocumentAssembler()
      .setInputCol("completions_raw")
      .setOutputCol("document")
    val entailment = model.setInputCols("document").setOutputCol("entailment")

    val pipeline =
      new org.apache.spark.ml.Pipeline().setStages(Array(documentAssembler, entailment))
    val annotations =
      com.johnsnowlabs.nlp.Annotation
        .collect(pipeline.fit(data).transform(data), "entailment")
        .head

    assert(annotations.length == samples.length)
    val carrying = annotations.filter(_.metadata.contains("entailment_matrix"))
    assert(
      carrying.length == 1,
      "the row-level matrix should be written once, not on every sample")

    val matrix =
      UncertaintyMetrics.parseEntailmentMatrix(carrying.head.metadata("entailment_matrix")).get
    assert(matrix.length == samples.length)
    matrix.indices.foreach(i => assert(math.abs(matrix(i)(i) - 1.0f) < 1e-6))
    // The two paraphrases should entail each other more than either entails the odd one out.
    assert(matrix(0)(1) > matrix(0)(2))
    assert(matrix(1)(0) > matrix(1)(2))
  }
}
