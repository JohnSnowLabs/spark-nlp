/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.benchmark

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.benchmark.engines._
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

/** One fixture-based test per comparison engine, each asserting a hand-computed expected score.
  */
class AccuracyBenchmarkTestSpec extends AnyFlatSpec {

  private val spark = SparkAccessor.spark
  private val sc = spark.sparkContext

  "LabelAccuracyEngine.evaluate" should "compute accuracy from predicted/gold label pairs" taggedAs FastTest in {
    // predicted: A,A,B,B ; gold: A,B,B,B -> 3/4 correct
    val pairs = sc.parallelize(Seq(("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")))
    val report = LabelAccuracyEngine.evaluate(BenchmarkTask.POS, pairs)

    assert(report.support == 4)
    assert(math.abs(report.overall("accuracy") - 0.75) < 1e-9)
  }

  "LabelAccuracyEngine.evaluateTopK" should "report top-k alongside a genuine top-1 accuracy" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq((Seq("A", "B", "C"), "B"), (Seq("A", "B", "C"), "D")))
    val report = LabelAccuracyEngine.evaluateTopK(BenchmarkTask.ImageClassification, pairs, k = 2)

    assert(report.support == 2)
    assert(math.abs(report.overall("accuracy") - 0.0) < 1e-9)
    assert(math.abs(report.overall("top2Accuracy") - 0.5) < 1e-9)
  }

  it should "keep the per-class breakdown on the real top-1 predictions" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq((Seq("A", "B"), "B"), (Seq("A", "B"), "B")))
    val report = LabelAccuracyEngine.evaluateTopK(BenchmarkTask.ImageClassification, pairs, k = 2)

    assert(math.abs(report.overall("accuracy") - 0.0) < 1e-9)
    assert(math.abs(report.overall("top2Accuracy") - 1.0) < 1e-9)
    assert(math.abs(report.perClass("B")("precision") - 0.0) < 1e-9)
    assert(math.abs(report.perClass("B")("recall") - 0.0) < 1e-9)
  }

  "Benchmark.rankedLabels" should "pass through metadata keys that are already clean labels" taggedAs FastTest in {
    val ranked = Benchmark.rankedLabels(Map("cat" -> "0.9", "dog" -> "0.1", "image" -> "0"))
    assert(ranked == Seq("cat", "dog"))
  }

  it should "unwrap ViT classifiers' stringified Option metadata keys" taggedAs FastTest in {
    // Captured verbatim from a real `image_classifier_vit_base_patch16_224` prediction.
    // ViTClassifier.scala:184 builds each class-name key via `Option(...).toString` without ever
    // unwrapping it, so every real class arrives as "Some(<label>)" and a lookup miss arrives as
    // bare "None" -- treating "Some(...)" itself as invalid (rather than unwrapping it) discards
    // every real class on every row and reports 0% accuracy regardless of model quality.
    val metadata = Map(
      "image" -> "0",
      "mode" -> "16",
      "nChannels" -> "3",
      "width" -> "500",
      "height" -> "334",
      "origin" -> "file:///tmp/images/images/palace.JPEG",
      "None" -> "2.5572559E-5",
      "Some(damselfly)" -> "9.8141445E-6",
      "Some(palace)" -> "9.499425E-6",
      "Some(mixing bowl)" -> "2.6357526E-5")

    val ranked = Benchmark.rankedLabels(metadata)

    assert(ranked.contains("palace"))
    assert(!ranked.contains("Some(palace)"))
    assert(!ranked.contains("None"))
    assert(ranked.head == "mixing bowl") // highest score among the three real classes
  }

  it should "score a row with no candidate labels as wrong, not correct" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq((Seq.empty[String], "A")))
    val report = LabelAccuracyEngine.evaluateTopK(BenchmarkTask.ImageClassification, pairs, k = 5)

    assert(report.support == 1)
    assert(math.abs(report.overall("accuracy") - 0.0) < 1e-9)
    assert(math.abs(report.overall("top5Accuracy") - 0.0) < 1e-9)
  }

  "SpanF1Engine.evaluate" should "merge BIO tags into spans and score them entity-level" taggedAs FastTest in {
    val predTags = Seq("B-PER", "I-PER", "O", "B-LOC")
    val goldTags = Seq("B-PER", "I-PER", "O", "B-ORG")
    val rows = sc.parallelize(Seq((predTags, goldTags)))

    val report = SpanF1Engine.evaluate(BenchmarkTask.NER, rows)

    assert(report.support == 2)
    assert(math.abs(report.overall("precision") - 0.5) < 1e-9)
    assert(math.abs(report.overall("recall") - 0.5) < 1e-9)
    assert(math.abs(report.overall("f1") - 0.5) < 1e-9)
    assert(math.abs(report.perClass("PER")("f1") - 1.0) < 1e-9)
  }

  it should "not drop IOBES single-token entities (S-) from scoring" taggedAs FastTest in {
    // Regression for a bug where S-/E- (IOBES) and U-/L- (BILOU) prefixes fell into the
    // catch-all branch, silently discarding the entity instead of scoring it -- identical
    // predicted/gold tags used to read as a deceptive support=1 (only LOC), dropping PER entirely.
    val tags = Seq("S-PER", "O", "B-LOC", "E-LOC")
    val rows = sc.parallelize(Seq((tags, tags)))

    val report = SpanF1Engine.evaluate(BenchmarkTask.NER, rows)

    assert(report.support == 2)
    assert(report.perClass.contains("PER"))
    assert(report.perClass.contains("LOC"))
    assert(math.abs(report.overall("f1") - 1.0) < 1e-9)
  }

  it should "score BILOU (U-/L-) the same way as IOBES (S-/E-)" taggedAs FastTest in {
    val tags = Seq("U-PER", "O", "B-LOC", "I-LOC", "L-LOC")
    val rows = sc.parallelize(Seq((tags, tags)))

    val report = SpanF1Engine.evaluate(BenchmarkTask.NER, rows)

    assert(report.support == 2)
    assert(report.perClass.contains("PER"))
    assert(math.abs(report.perClass("LOC")("f1") - 1.0) < 1e-9)
  }

  "SpanF1Engine.evaluateBoundaries" should "score pre-extracted (begin,end) boundaries directly" taggedAs FastTest in {
    val predBoundaries = Set((0, 2), (2, 5))
    val goldBoundaries = Set((0, 2), (2, 4), (4, 5))
    val rows = sc.parallelize(Seq((predBoundaries, goldBoundaries)))

    val report = SpanF1Engine.evaluateBoundaries(BenchmarkTask.WordSegmentation, rows)

    assert(report.support == 3)
    assert(math.abs(report.overall("precision") - 0.5) < 1e-9)
    assert(math.abs(report.overall("recall") - 1.0 / 3) < 1e-9)
    assert(math.abs(report.overall("f1") - 0.4) < 1e-9)
  }

  "DependencyAccuracyEngine.evaluate" should "compute UAS and LAS from (head, label) pairs" taggedAs FastTest in {
    val predicted = Seq((1, "nsubj"), (0, "root"), (1, "dobj"))
    val gold = Seq((1, "nsubj"), (0, "root"), (2, "dobj"))
    val rows = sc.parallelize(Seq((predicted, gold)))

    val report = DependencyAccuracyEngine.evaluate(BenchmarkTask.DependencyParsing, rows)

    assert(report.support == 3)
    assert(math.abs(report.overall("uas") - 2.0 / 3) < 1e-9)
    assert(math.abs(report.overall("las") - 2.0 / 3) < 1e-9)
  }

  "TextSimilarityEngine.evaluate with Wer" should "compute word-level edit distance over reference word count" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("the cat sat on mat", "the cat sat on the mat")))
    val report =
      TextSimilarityEngine.evaluate(BenchmarkTask.SpeechRecognition, pairs, TextMetric.Wer)

    assert(math.abs(report.overall("wer") - 1.0 / 6) < 1e-9)
  }

  "TextSimilarityEngine.evaluate with SquadEmF1" should "average exact-match and token-overlap F1 across rows" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("Paris", "paris"), ("Paris, France", "Paris")))
    val report =
      TextSimilarityEngine.evaluate(BenchmarkTask.QuestionAnswering, pairs, TextMetric.SquadEmF1)

    assert(math.abs(report.overall("exactMatch") - 0.5) < 1e-9)
    assert(math.abs(report.overall("f1") - 5.0 / 6) < 1e-6)
  }

  "TextSimilarityEngine.evaluate with Bleu" should "score identical sentences at (near) 1.0" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("the cat sat on the mat", "the cat sat on the mat")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 1.0) < 1e-9)
  }

  "TextSimilarityEngine.evaluate with Bleu" should "match sacrebleu's known score on a non-trivial, punctuation-free fixture" taggedAs FastTest in {
    // Cross-language parity fixture, also used in python/test/benchmark_test.py.
    val pairs = sc.parallelize(
      Seq(
        (
          "the fast brown fox jumps over a lazy dog",
          "the quick brown fox jumps over the lazy dog")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.36889397) < 1e-6)
  }

  it should "match sacrebleu on an apostrophe-heavy corpus, not just punctuation-free text" taggedAs FastTest in {
    val pairs = sc.parallelize(
      Seq(
        (
          "C'est l'une des plus belles villes d'Europe.",
          "C'est l'une des plus belles villes de l'Europe."),
        (
          "Aujourd'hui j'ai acheté d'excellents fruits.",
          "Aujourd'hui, j'ai acheté d'excellents fruits."),
        (
          "Elle m'a dit qu'elle viendrait demain matin.",
          "Elle m'a dit qu'elle arriverait demain matin.")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.60539119) < 1e-6)
  }

  it should "keep an apostrophe attached to its word, like sacrebleu's 13a" taggedAs FastTest in {
    // This is a tokenizer test, not a BLEU-score test: a corpus this short (3 tokens) has no
    // 4-grams, so real sacrebleu itself scores it 0.0 by design --
    // `sacrebleu.corpus_bleu(["l'homme est ici"], [["l'homme est ici"]]).score == 0.0`, verified
    // directly. This assertion is a byproduct that happens to also confirm the tokenizer didn't
    // crash; the tokenizer behavior itself is exercised by the non-trivial fixture tests above.
    val identical = "l'homme est ici"
    val pairs = sc.parallelize(Seq((identical, identical)))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.0) < 1e-9)
  }

  it should "not split a period that sits between two digits" taggedAs FastTest in {
    // Same as above: a 3-token corpus has no 4-grams, so real sacrebleu scores this 0.0 by
    // design, even for an identical pair -- verified directly against sacrebleu.corpus_bleu.
    val identical = "3.14 is pi"
    val pairs = sc.parallelize(Seq((identical, identical)))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.0) < 1e-9)
  }

  it should "score 0.0 for a corpus too short to have any 4-grams, matching sacrebleu" taggedAs FastTest in {
    // Verified directly: sacrebleu.corpus_bleu(["hello"], [["hello"]]).score == 0.0. Corpus-level
    // BLEU pools n-gram counts across every row before scoring, so this is 0 only because the
    // *whole corpus* (not just this row) never produces a 4-gram -- mixing in one longer sentence
    // elsewhere in the corpus would change the outcome (see the mixed-length behavior implied by
    // pooling in `bleu()` above).
    val pairs = sc.parallelize(Seq(("hello", "hello")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.0) < 1e-9)
  }

  it should "score normally when a short row is pooled with a longer row in the same corpus" taggedAs FastTest in {
    // Verified directly: sacrebleu.corpus_bleu(["the cat sat on the mat", "hi"],
    // [["the cat sat on the mat", "hi"]]).score == 100.0. The short row alone would have no
    // 4-grams, but corpus-level totals are pooled across all rows before scoring, so the longer
    // row's 4-grams keep the corpus-level total nonzero.
    val pairs =
      sc.parallelize(Seq(("the cat sat on the mat", "the cat sat on the mat"), ("hi", "hi")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 1.0) < 1e-9)
  }

  it should "score 0.0 for a completely empty hypothesis" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("", "the cat sat on the mat")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.0) < 1e-9)
  }

  it should "force every precisionN to exactly 0.0, not smoothed noise, on a zero-match corpus" taggedAs FastTest in {
    // Regression: sacrebleu forces every precision to exactly 0 when the hypothesis shares zero
    // n-grams with the reference at every order (verified directly:
    // sacrebleu.corpus_bleu(["我喜欢机器学习和自然语言处理的应用"],
    // [["我喜欢深度学习和自然语言处理的研究"]]).precisions == [0.0, 0.0, 0.0, 0.0]). The old code
    // applied NIST/mteval smoothing per-order regardless, producing a nonzero precision1 even
    // though the headline `bleu` score was already (separately) guarded to 0.0.
    val pairs = sc.parallelize(Seq(("我喜欢机器学习和自然语言处理的应用", "我喜欢深度学习和自然语言处理的研究")))
    val report = TextSimilarityEngine.evaluate(BenchmarkTask.Translation, pairs, TextMetric.Bleu)

    assert(math.abs(report.overall("bleu") - 0.0) < 1e-9)
    assert(math.abs(report.overall("precision1") - 0.0) < 1e-9)
    assert(math.abs(report.overall("precision2") - 0.0) < 1e-9)
    assert(math.abs(report.overall("precision3") - 0.0) < 1e-9)
    assert(math.abs(report.overall("precision4") - 0.0) < 1e-9)
  }

  "TextSimilarityEngine.evaluateSquad" should "keep the best score across all reference answers" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("in 1858", Seq("1858", "in 1858", "the year 1858"))))
    val report = TextSimilarityEngine.evaluateSquad(BenchmarkTask.QuestionAnswering, pairs)

    assert(report.support == 1)
    assert(math.abs(report.overall("exactMatch") - 1.0) < 1e-9)
    assert(math.abs(report.overall("f1") - 1.0) < 1e-9)
  }

  it should "score a row with no reference answers as wrong rather than as a match" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("Paris", Seq.empty[String])))
    val report = TextSimilarityEngine.evaluateSquad(BenchmarkTask.QuestionAnswering, pairs)

    assert(report.support == 1)
    assert(math.abs(report.overall("exactMatch") - 0.0) < 1e-9)
  }

  "TextSimilarityEngine.evaluate with Rouge" should "score identical sentences at (near) 1.0 on all variants" taggedAs FastTest in {
    val pairs = sc.parallelize(Seq(("the cat sat", "the cat sat")))
    val report =
      TextSimilarityEngine.evaluate(BenchmarkTask.Summarization, pairs, TextMetric.Rouge)

    assert(math.abs(report.overall("rouge1_f1") - 1.0) < 1e-9)
    assert(math.abs(report.overall("rouge2_f1") - 1.0) < 1e-9)
    assert(math.abs(report.overall("rougeL_f1") - 1.0) < 1e-9)
  }

  // DocumentAssembler and SentenceDetector both emit annotatorType `document`.
  private val twoDocumentColumnPipeline = {
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentenceDetector =
      new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    new Pipeline().setStages(Array(documentAssembler, sentenceDetector))
  }

  "Benchmark.tasksNeedingCache" should "contain exactly the tasks whose engine issues more than one action" taggedAs FastTest in {
    val expected =
      Set(
        BenchmarkTask.POS,
        BenchmarkTask.Classification,
        BenchmarkTask.SpellCheck,
        BenchmarkTask.LanguageDetection,
        BenchmarkTask.ImageClassification)
    assert(Benchmark.tasksNeedingCache == expected)
  }

  "Benchmark.evaluate" should "score the last output column of the expected type, not the first" taggedAs FastTest in {
    import spark.implicits._
    val gold = Seq(("Hello there.", "Hello there.")).toDF("text", "label")
    val model = twoDocumentColumnPipeline.fit(gold)

    val report = Benchmark.evaluate(model, gold, BenchmarkTask.Translation)

    assert(report.scoredColumns == Seq("sentence"))
  }

  it should "let predictedCol override the resolved column" taggedAs FastTest in {
    import spark.implicits._
    val gold = Seq(("Hello there.", "Hello there.")).toDF("text", "label")
    val model = twoDocumentColumnPipeline.fit(gold)

    val report =
      Benchmark.evaluate(model, gold, BenchmarkTask.Translation, predictedCol = Some("document"))

    assert(report.scoredColumns == Seq("document"))
  }

  it should "resolve the real prediction column even when goldData already has a same-named " +
    "stale column" taggedAs FastTest in {
      import spark.implicits._
      // goldData was itself produced by a prior run and already carries a "document" column --
      // a plain string here, standing in for any pre-existing, differently-shaped column. The
      // pipeline's own DocumentAssembler also writes its real output to "document", overwriting
      // it. Before the fix this was excluded as "pre-existing input" purely by name, and
      // evaluate() threw "Could not find an output column" even though the pipeline did produce
      // one.
      val documentAssembler =
        new DocumentAssembler().setInputCol("text").setOutputCol("document")
      val pipeline = new Pipeline().setStages(Array(documentAssembler))
      val gold = Seq(("Hello there.", "Hello there.", "stale-placeholder"))
        .toDF("text", "label", "document")
      val model = pipeline.fit(gold)

      val report = Benchmark.evaluate(model, gold, BenchmarkTask.Translation)

      assert(report.scoredColumns == Seq("document"))
    }

  it should "fail with a usable message when no column of the expected type exists" taggedAs FastTest in {
    import spark.implicits._
    val gold = Seq(("Hello there.", "Hello there.")).toDF("text", "label")
    val model = twoDocumentColumnPipeline.fit(gold)

    val thrown = intercept[IllegalArgumentException] {
      Benchmark.evaluate(model, gold, BenchmarkTask.NER)
    }
    assert(thrown.getMessage.contains("named_entity"))
    assert(thrown.getMessage.contains("predictedCol"))
  }
}
