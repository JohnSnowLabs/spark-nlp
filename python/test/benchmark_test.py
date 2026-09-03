#  Copyright 2017-2024 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""One fixture-based test per comparison engine, mirroring AccuracyBenchmarkTestSpec.scala."""
import unittest

import pytest

from pyspark.sql.types import StringType, StructField, StructType

from sparknlp.benchmark import (
    _bleu,
    _dependency_accuracy,
    _find_output_column,
    _label_accuracy,
    _label_accuracy_top_k,
    _ranked_labels,
    _rouge,
    _span_f1,
    _span_f1_boundaries,
    _squad_em_f1,
    _wer,
)
from test.util import SparkContextForTest


@pytest.mark.fast
class LabelAccuracyEngineTestSpec(unittest.TestCase):

    def setUp(self):
        self.sc = SparkContextForTest.spark.sparkContext

    def test_evaluate_computes_accuracy(self):
        pairs = self.sc.parallelize([("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")])
        report = _label_accuracy("pos", pairs)

        self.assertEqual(report.support, 4)
        self.assertAlmostEqual(report.overall["accuracy"], 0.75)

    def test_top_k_reports_top_k_alongside_a_genuine_top_1_accuracy(self):
        pairs = self.sc.parallelize([(["A", "B", "C"], "B"), (["A", "B", "C"], "D")])
        report = _label_accuracy_top_k("imageclassification", pairs, k=2)

        self.assertEqual(report.support, 2)
        self.assertAlmostEqual(report.overall["accuracy"], 0.0)
        self.assertAlmostEqual(report.overall["top2Accuracy"], 0.5)

    def test_top_k_keeps_per_class_on_the_real_top_1_predictions(self):
        pairs = self.sc.parallelize([(["A", "B"], "B"), (["A", "B"], "B")])
        report = _label_accuracy_top_k("imageclassification", pairs, k=2)

        self.assertAlmostEqual(report.overall["accuracy"], 0.0)
        self.assertAlmostEqual(report.overall["top2Accuracy"], 1.0)
        self.assertAlmostEqual(report.per_class["B"]["precision"], 0.0)
        self.assertAlmostEqual(report.per_class["B"]["recall"], 0.0)

    def test_top_k_scores_no_candidate_labels_as_wrong_not_correct(self):
        pairs = self.sc.parallelize([([], "A")])
        report = _label_accuracy_top_k("imageclassification", pairs, k=5)

        self.assertEqual(report.support, 1)
        self.assertAlmostEqual(report.overall["accuracy"], 0.0)
        self.assertAlmostEqual(report.overall["top5Accuracy"], 0.0)

    def test_ranked_labels_passes_through_keys_that_are_already_clean(self):
        ranked = _ranked_labels({"cat": "0.9", "dog": "0.1", "image": "0"})
        self.assertEqual(ranked, ["cat", "dog"])

    def test_ranked_labels_unwraps_vit_classifiers_stringified_option_keys(self):
        # Captured verbatim from a real `image_classifier_vit_base_patch16_224` prediction.
        # ViTClassifier.scala:184 builds each class-name key via `Option(...).toString` without
        # ever unwrapping it, so every real class arrives as "Some(<label>)" and a lookup miss
        # arrives as bare "None" -- treating "Some(...)" itself as invalid (rather than
        # unwrapping it) discards every real class on every row and reports 0% accuracy
        # regardless of model quality.
        metadata = {
            "image": "0", "mode": "16", "nChannels": "3", "width": "500", "height": "334",
            "origin": "file:///tmp/images/images/palace.JPEG",
            "None": "2.5572559E-5",
            "Some(damselfly)": "9.8141445E-6",
            "Some(palace)": "9.499425E-6",
            "Some(mixing bowl)": "2.6357526E-5",
        }
        ranked = _ranked_labels(metadata)

        self.assertIn("palace", ranked)
        self.assertNotIn("Some(palace)", ranked)
        self.assertNotIn("None", ranked)
        self.assertEqual(ranked[0], "mixing bowl")  # highest score among the three real classes


@pytest.mark.fast
class SpanF1EngineTestSpec(unittest.TestCase):

    def setUp(self):
        self.sc = SparkContextForTest.spark.sparkContext

    def test_evaluate_merges_bio_tags_into_spans(self):
        pred_tags = ["B-PER", "I-PER", "O", "B-LOC"]
        gold_tags = ["B-PER", "I-PER", "O", "B-ORG"]
        report = _span_f1("ner", self.sc.parallelize([(pred_tags, gold_tags)]))

        self.assertEqual(report.support, 2)
        self.assertAlmostEqual(report.overall["precision"], 0.5)
        self.assertAlmostEqual(report.overall["recall"], 0.5)
        self.assertAlmostEqual(report.overall["f1"], 0.5)
        self.assertAlmostEqual(report.per_class["PER"]["f1"], 1.0)

    def test_evaluate_boundaries_scores_pre_extracted_spans(self):
        pred_boundaries = {(0, 2), (2, 5)}
        gold_boundaries = {(0, 2), (2, 4), (4, 5)}
        report = _span_f1_boundaries(
            "wordsegmentation", self.sc.parallelize([(pred_boundaries, gold_boundaries)]))

        self.assertEqual(report.support, 3)
        self.assertAlmostEqual(report.overall["precision"], 0.5)
        self.assertAlmostEqual(report.overall["recall"], 1.0 / 3)
        self.assertAlmostEqual(report.overall["f1"], 0.4)

    def test_does_not_drop_iobes_single_token_entities(self):
        # Regression: S-/E- (IOBES) and U-/L- (BILOU) prefixes used to fall into the catch-all
        # branch, silently discarding the entity -- identical predicted/gold tags used to read as
        # a deceptive support=1 (only LOC), dropping PER entirely.
        tags = ["S-PER", "O", "B-LOC", "E-LOC"]
        report = _span_f1("ner", self.sc.parallelize([(tags, tags)]))

        self.assertEqual(report.support, 2)
        self.assertIn("PER", report.per_class)
        self.assertIn("LOC", report.per_class)
        self.assertAlmostEqual(report.overall["f1"], 1.0)

    def test_scores_bilou_the_same_way_as_iobes(self):
        tags = ["U-PER", "O", "B-LOC", "I-LOC", "L-LOC"]
        report = _span_f1("ner", self.sc.parallelize([(tags, tags)]))

        self.assertEqual(report.support, 2)
        self.assertIn("PER", report.per_class)
        self.assertAlmostEqual(report.per_class["LOC"]["f1"], 1.0)


@pytest.mark.fast
class DependencyAccuracyEngineTestSpec(unittest.TestCase):

    def setUp(self):
        self.sc = SparkContextForTest.spark.sparkContext

    def test_evaluate_computes_uas_and_las(self):
        predicted = [(1, "nsubj"), (0, "root"), (1, "dobj")]
        gold = [(1, "nsubj"), (0, "root"), (2, "dobj")]
        report = _dependency_accuracy("dependencyparsing", self.sc.parallelize([(predicted, gold)]))

        self.assertEqual(report.support, 3)
        self.assertAlmostEqual(report.overall["uas"], 2.0 / 3)
        self.assertAlmostEqual(report.overall["las"], 2.0 / 3)

    def test_evaluate_raises_on_predicted_gold_length_mismatch(self):
        predicted = [(1, "nsubj"), (0, "root")]
        gold = [(1, "nsubj"), (0, "root"), (2, "dobj")]
        with self.assertRaises(Exception):
            _dependency_accuracy("dependencyparsing", self.sc.parallelize([(predicted, gold)]))


@pytest.mark.fast
class TextSimilarityEngineTestSpec(unittest.TestCase):

    def setUp(self):
        self.sc = SparkContextForTest.spark.sparkContext

    def test_squad_em_f1_averages_across_rows(self):
        pairs = self.sc.parallelize([("Paris", ["paris"]), ("Paris, France", ["Paris"])])
        overall, support = _squad_em_f1(pairs)

        self.assertEqual(support, 2)
        self.assertAlmostEqual(overall["exactMatch"], 0.5)
        self.assertAlmostEqual(overall["f1"], 5.0 / 6, places=6)

    def test_squad_em_f1_keeps_the_best_score_across_reference_answers(self):
        pairs = self.sc.parallelize([("in 1858", ["1858", "in 1858", "the year 1858"])])
        overall, support = _squad_em_f1(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["exactMatch"], 1.0)
        self.assertAlmostEqual(overall["f1"], 1.0)

    def test_squad_em_f1_scores_a_row_with_no_references_as_wrong(self):
        pairs = self.sc.parallelize([("Paris", [])])
        overall, support = _squad_em_f1(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["exactMatch"], 0.0)

    def test_wer_matches_scala_side_computation(self):
        pytest.importorskip("jiwer")
        pairs = self.sc.parallelize([("the cat sat on mat", "the cat sat on the mat")])
        overall, support = _wer(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["wer"], 1.0 / 6)

    def test_wer_matches_jiwer_itself_on_a_whitespace_dirty_reference(self):
        # Regression: the numerator (edits) used to come from jiwer's own tokenization of `gold`
        # while the denominator came from a separate `gold.split()` -- inconsistent whenever
        # `gold` has a tab/embedded newline, since jiwer's default pipeline only splits on plain
        # spaces. Verified directly against jiwer: jiwer.process_words("the\tcat sat  on the
        # mat", "the cat sat on the mat").wer == 0.4 (edits=2 over 5 jiwer-tokenized words, since
        # "the\tcat" counts as one word to jiwer).
        pytest.importorskip("jiwer")
        gold = "the\tcat sat  on the mat"
        pred = "the cat sat on the mat"
        pairs = self.sc.parallelize([(pred, gold)])
        overall, support = _wer(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["wer"], 0.4)

    def test_bleu_scores_identical_sentences_near_one(self):
        pytest.importorskip("sacrebleu")
        pairs = self.sc.parallelize([("the cat sat on the mat", "the cat sat on the mat")])
        overall, support = _bleu(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["bleu"], 1.0, places=6)

    def test_bleu_matches_sacrebleus_known_score(self):
        # Cross-language parity fixture, also used in AccuracyBenchmarkTestSpec.scala.
        pytest.importorskip("sacrebleu")
        pairs = self.sc.parallelize([
            ("the fast brown fox jumps over a lazy dog",
             "the quick brown fox jumps over the lazy dog")])
        overall, support = _bleu(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["bleu"], 0.36889397, places=6)

    def test_bleu_matches_sacrebleu_on_an_apostrophe_heavy_corpus(self):
        pytest.importorskip("sacrebleu")
        pairs = self.sc.parallelize([
            ("C'est l'une des plus belles villes d'Europe.",
             "C'est l'une des plus belles villes de l'Europe."),
            ("Aujourd'hui j'ai acheté d'excellents fruits.",
             "Aujourd'hui, j'ai acheté d'excellents fruits."),
            ("Elle m'a dit qu'elle viendrait demain matin.",
             "Elle m'a dit qu'elle arriverait demain matin."),
        ])
        overall, support = _bleu(pairs)

        self.assertEqual(support, 3)
        self.assertAlmostEqual(overall["bleu"], 0.60539119, places=6)

    def test_bleu_scores_zero_for_a_corpus_too_short_to_have_any_4grams(self):
        # Cross-language parity fixture, mirroring AccuracyBenchmarkTestSpec.scala. Verified
        # directly against sacrebleu: a 3-token sentence has no 4-grams, so corpus-level BLEU is
        # 0.0 by design -- sacrebleu.corpus_bleu(["3.14 is pi"], [["3.14 is pi"]]).score == 0.0 --
        # even for an otherwise-perfect match. This is standard BLEU behavior, not a bug.
        pytest.importorskip("sacrebleu")
        pairs = self.sc.parallelize([("3.14 is pi", "3.14 is pi")])
        overall, support = _bleu(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["bleu"], 0.0)

    def test_bleu_scores_normally_when_a_short_row_is_pooled_with_a_longer_row(self):
        # Cross-language parity fixture, mirroring AccuracyBenchmarkTestSpec.scala. Corpus-level
        # BLEU pools n-gram counts across every row before scoring, so the short row above only
        # scores 0.0 because the *whole* corpus lacks a 4-gram -- mixing in one longer row changes
        # that. Verified directly: sacrebleu.corpus_bleu(
        #   ["the cat sat on the mat", "hi"], [["the cat sat on the mat", "hi"]]).score == 100.0.
        pytest.importorskip("sacrebleu")
        pairs = self.sc.parallelize([
            ("the cat sat on the mat", "the cat sat on the mat"),
            ("hi", "hi"),
        ])
        overall, support = _bleu(pairs)

        self.assertEqual(support, 2)
        self.assertAlmostEqual(overall["bleu"], 1.0)

    def test_rouge_scores_identical_sentences_near_one(self):
        pytest.importorskip("rouge_score")
        pairs = self.sc.parallelize([("the cat sat", "the cat sat")])
        overall, support = _rouge(pairs)

        self.assertEqual(support, 1)
        self.assertAlmostEqual(overall["rouge1_f1"], 1.0)
        self.assertAlmostEqual(overall["rouge2_f1"], 1.0)
        self.assertAlmostEqual(overall["rougeL_f1"], 1.0)


@pytest.mark.fast
class ColumnResolutionTestSpec(unittest.TestCase):

    def setUp(self):
        self.spark = SparkContextForTest.spark

    def test_treats_a_same_named_overwritten_column_as_newly_produced(self):
        # gold_data was itself produced by a prior run and already carries a "document" column --
        # a plain string here, standing in for any pre-existing, differently-shaped column. The
        # pipeline's real output also lands under "document", overwriting it with an
        # annotator-typed column. Before the fix this was excluded as "pre-existing input" purely
        # by name, and _find_output_column raised even though the pipeline did produce one.
        gold_schema = StructType([
            StructField("text", StringType(), True),
            StructField("label", StringType(), True),
            StructField("document", StringType(), True),
        ])
        transformed_schema = StructType([
            StructField("text", StringType(), True),
            StructField("label", StringType(), True),
            StructField("document", StringType(), True, {"annotatorType": "document"}),
        ])
        gold_data = self.spark.createDataFrame([("hi", "hi", "stale-placeholder")], gold_schema)
        transformed = self.spark.createDataFrame([("hi", "hi", "real-output")], transformed_schema)

        name = _find_output_column(
            "translation", "document", gold_data.schema.fields, transformed, None)

        self.assertEqual(name, "document")

    def test_still_excludes_an_untouched_passthrough_column(self):
        # A column that is genuinely unchanged (same name, type, and metadata) must still be
        # excluded, so the "last matching column wins" rule doesn't pick up plain input columns.
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("label", StringType(), True),
            StructField("document", StringType(), True, {"annotatorType": "document"}),
        ])
        gold_data = self.spark.createDataFrame([("hi", "hi", "unchanged")], schema)
        transformed = self.spark.createDataFrame([("hi", "hi", "unchanged")], schema)

        with self.assertRaises(ValueError):
            _find_output_column("translation", "document", gold_data.schema.fields, transformed, None)
