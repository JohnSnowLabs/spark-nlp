#  Copyright 2017-2022 John Snow Labs
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
"""Throughput and accuracy benchmarking for a fitted PipelineModel, against data supplied by
the caller. See ``examples/python/benchmarks/`` for end-to-end recipes with recommended public
datasets per task.

Mirrors ``com.johnsnowlabs.nlp.benchmark.Benchmark`` on the Scala side.
"""

import math
import re
from collections import Counter

import pyspark.sql.functions as F

SUPPORTED_TASKS = {
    "ner", "wordsegmentation", "pos", "classification", "spellcheck", "languagedetection",
    "imageclassification", "dependencyparsing", "questionanswering", "speechrecognition",
    "translation", "summarization",
}

# Tasks whose engine issues more than one action against the transformed DataFrame; only these
# benefit from caching it.
_TASKS_NEEDING_CACHE = {"pos", "classification", "spellcheck", "languagedetection",
                        "imageclassification"}

_TEXT_SIMILARITY_METRIC = {
    "questionanswering": "squad",
    "speechrecognition": "wer",
    "translation": "bleu",
    "summarization": "rouge",
}

_ANNOTATOR_TYPE = {
    "ner": "named_entity",
    "wordsegmentation": "token",
    "pos": "pos",
    "classification": "category",
    "spellcheck": "token",
    "languagedetection": "language",
    "imageclassification": "category",
}

# Sentinel for a row with no usable prediction, so it never accidentally matches a real gold
# label.
_NO_PREDICTION = "<<no_prediction>>"

#: Cap on how many per-class lines ``AccuracyReport.__repr__`` prints before eliding.
MAX_PRINTED_CLASSES = 50


class MetricRate:
    """Throughput of a single output annotator type, averaged over the timed trials."""

    def __init__(self, annotator_type, output_column, total_items, mean_items_per_second,
                 confidence_interval_95):
        self.annotator_type = annotator_type
        self.output_column = output_column
        self.total_items = total_items
        self.mean_items_per_second = mean_items_per_second
        self.confidence_interval_95 = confidence_interval_95

    def __repr__(self):
        return (f"{self.output_column:<20} {self.mean_items_per_second:,.1f} ± "
                f"{self.confidence_interval_95:,.1f} items/sec "
                f"(type: {self.annotator_type}, n={self.total_items})")


class ThroughputReport:
    """Result of :func:`Benchmark.throughput`: one :class:`MetricRate` per annotation type the
    pipeline produced, plus the raw per-trial elapsed times behind them."""

    def __init__(self, rates, trial_seconds):
        self.rates = rates
        self.trial_seconds = trial_seconds

    def __repr__(self):
        mean_elapsed = sum(self.trial_seconds) / len(self.trial_seconds)
        header = (f"Throughput over {len(self.trial_seconds)} trial(s), "
                  f"mean elapsed {mean_elapsed:,.3f} sec/trial")
        lines = [header] + [f"  {r}" for r in self.rates]
        return "\n".join(lines)


class AccuracyReport:
    """Result of :func:`Benchmark.evaluate`. The metric names in ``overall``/``per_class``
    depend on the task: label-accuracy tasks report accuracy/weightedPrecision/weightedRecall/
    weightedF1 (support-weighted across labels); NER and word segmentation report entity/
    segment-level precision/recall/f1 (micro-averaged: pooled counts across every type, matching
    conlleval/seqeval, via exact span-boundary matching); dependency parsing reports uas/las; the
    text-similarity tasks report their own metric (bleu, rouge1/rouge2/rougeL, wer, or
    exact_match/f1).

    These averaging conventions differ by task and are NOT interchangeable: ``overall["f1"]``
    from a classification report and from a NER report are computed differently and are not
    comparable to each other. They can also differ from a metric the same model already reported
    during its own training-time evaluation (which typically uses per-token tag accuracy for NER,
    or macro-averaged F1 for classification) -- that is expected, not a discrepancy to reconcile.

    ``scored_columns`` names which of the pipeline's output columns the score was computed from.
    """

    def __init__(self, task, overall, per_class=None, support=0, scored_columns=None):
        self.task = task
        self.overall = overall
        self.per_class = per_class or {}
        self.support = support
        self.scored_columns = list(scored_columns or [])

    def __repr__(self):
        overall_line = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.overall.items()))
        scored = f", scored: {', '.join(self.scored_columns)}" if self.scored_columns else ""
        header = f"{self.task} accuracy (n={self.support}{scored}): {overall_line}"
        if not self.per_class:
            return header
        lines = [header]
        ordered = sorted(self.per_class.items())
        for label, metrics in ordered[:MAX_PRINTED_CLASSES]:
            m = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            lines.append(f"  {label}: {m}")
        if len(ordered) > MAX_PRINTED_CLASSES:
            lines.append(
                f"  ... and {len(ordered) - MAX_PRINTED_CLASSES} more labels "
                "(see the per_class attribute for the full breakdown)")
        return "\n".join(lines)


def _confidence_interval_95(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return 1.96 * math.sqrt(variance / len(values))


def _newly_produced_fields(input_fields, transformed_fields):
    """Fields in `transformed_fields` that don't already exist, unchanged, in `input_fields`.

    Matches on the whole field (name, type, and metadata), not just the name: a caller can pass
    gold data that was itself produced by an earlier Spark NLP run and already carries a
    same-named column (e.g. "category" or "document"). If the new pipeline's real prediction also
    lands under that name, transform() overwrites it with a differently-typed/annotated field --
    matching by name alone would wrongly treat that fresh prediction as pre-existing input and
    exclude it.
    """
    return [f for f in transformed_fields if f not in input_fields]


def _columns_by_type(fields, exclude_fields):
    result = []
    for f in _newly_produced_fields(exclude_fields, fields):
        annotator_type = f.metadata.get("annotatorType") if f.metadata else None
        if annotator_type:
            result.append((f.name, annotator_type))
    return result


def _find_output_column(task, annotator_type, gold_input_fields, transformed, predicted_col,
                       override_param_name="predicted_col"):
    """Resolve which output column to score, taking the last match rather than the first --
    schema order is stage order, and the prediction is the last stage to write that type."""
    if predicted_col is not None:
        return predicted_col
    candidates = [name for name, t in _columns_by_type(transformed.schema.fields, gold_input_fields)
                  if t == annotator_type]
    if not candidates:
        raise ValueError(
            f"Could not find an output column of type '{annotator_type}' produced by this "
            f"pipeline for task '{task}'. Check that the pipeline actually produces this "
            f"annotation, or pass an explicit {override_param_name}.")
    return candidates[-1]


def _single_result_expr(annotation_col, on_missing):
    """SQL expression for the first annotation's `result` field, guarded against ANSI mode
    throwing on an empty array."""
    escaped = on_missing.replace("'", "''")
    return (f"COALESCE(IF(size(`{annotation_col}`) > 0, "
            f"element_at(`{annotation_col}`, 1).result, NULL), '{escaped}')")


def _zip_equal_length(task, pred, gold):
    if len(pred) != len(gold):
        raise ValueError(
            f"Benchmark.evaluate(task='{task}'): predicted and gold sequences have different "
            f"lengths ({len(pred)} vs {len(gold)}) for one row. label_col must align one-to-one "
            "with the pipeline's own tokenization for this task.")
    return list(zip(pred, gold))


class Benchmark:
    """Throughput and accuracy benchmarking for any fitted PipelineModel, against data
    supplied by the caller.

    Examples
    --------
    >>> from sparknlp.benchmark import Benchmark
    >>> report = Benchmark.throughput(pipeline_model, my_data)
    >>> print(report)
    >>> accuracy = Benchmark.evaluate(pipeline_model, my_gold_data, task="ner")
    >>> print(accuracy)
    """

    @staticmethod
    def throughput(pipeline_model, data, text_col="text", warmup_runs=1, trials=5):
        """Measures how fast ``pipeline_model`` processes ``data``, reporting one rate per
        type of annotation it produces. Runs ``warmup_runs`` untimed passes first, then
        ``trials`` timed passes, reporting the mean rate per metric with a 95% confidence
        interval.

        ``data`` is read once per pass (1 + ``warmup_runs`` + ``trials`` times in total), so if
        it isn't already cached this persists it for the duration of the call (and unpersists it
        again afterwards) -- otherwise a ``data`` sourced from an expensive or non-deterministic
        upstream read (a file scan, a ``.sample()``) would have that cost repeated on every pass,
        contaminating the measured rate and, for a non-deterministic source, varying the row
        count between trials.

        ``text_col`` is not read here -- the pipeline's own first stage decides which column it
        consumes.
        """
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be >= 0")
        if trials < 1:
            raise ValueError("trials must be >= 1")

        import time

        from pyspark import StorageLevel

        # Only persist here if the caller hasn't already -- leaves caller-managed caching
        # untouched instead of unpersisting a DataFrame we didn't cache ourselves.
        already_cached = data.storageLevel != StorageLevel.NONE
        if not already_cached:
            data.persist()
        try:
            produced_types = _columns_by_type(pipeline_model.transform(data).schema.fields, data.schema.fields)
            if not produced_types:
                raise ValueError("The pipeline did not add any annotator-typed output columns to the input data.")

            def count_cols():
                return [F.sum(F.size(F.col(name))).alias(name) for name, _ in produced_types]

            # Must run the same aggregation as the timed trials -- `.count()` alone gets
            # column-pruned and never executes the annotators.
            for _ in range(warmup_runs):
                pipeline_model.transform(data).select(*count_cols()).collect()

            trial_results = []
            for _ in range(trials):
                start = time.perf_counter()
                out = pipeline_model.transform(data)
                row = out.select(*count_cols()).collect()[0]
                elapsed = time.perf_counter() - start
                counts = [row[i] or 0 for i in range(len(produced_types))]
                trial_results.append((elapsed, counts))

            rates = []
            for i, (name, annotator_type) in enumerate(produced_types):
                per_trial_rates = [counts[i] / elapsed for elapsed, counts in trial_results]
                mean = sum(per_trial_rates) / len(per_trial_rates)
                total_items = sum(counts[i] for _, counts in trial_results)
                rates.append(MetricRate(annotator_type, name, total_items, mean,
                                         _confidence_interval_95(per_trial_rates)))

            return ThroughputReport(rates, [e for e, _ in trial_results])
        finally:
            if not already_cached:
                data.unpersist()

    @staticmethod
    def evaluate(pipeline_model, gold_data, task="ner", text_col="text", label_col="label",
                 predicted_col=None, predicted_labeled_dependency_col=None, top_k=5):
        """Scores ``pipeline_model``'s predictions on ``gold_data`` against ``task``'s
        accuracy metric.

        The expected shape of ``gold_data`` depends on ``task``:

        - ``ner``, ``pos``, ``wordsegmentation``: ``text_col`` (raw text) + ``label_col``
          (array<string> of gold tags/segment boundaries, aligned by position to the
          pipeline's own tokenization -- for ``wordsegmentation`` each entry is a
          ``"begin:end"`` character-offset string, inclusive on both ends)
        - ``classification``, ``spellcheck``, ``languagedetection``: ``text_col`` +
          ``label_col`` (a single gold label string per row)
        - ``imageclassification``: ``text_col`` (the pipeline's image input column, see note
          below) + ``label_col`` (single gold class label per row). Reports ``accuracy`` for the
          model's top-1 label like every other task, plus an extra ``top{top_k}Accuracy`` metric.
        - ``dependencyparsing``: ``text_col`` + ``label_col`` (array<string> of
          ``"headIndex:label"`` per token)
        - ``questionanswering``: ``text_col`` + ``label_col``, either a single reference answer
          per row or an ``array<string>`` of every acceptable answer for that question, scored
          against its best-matching reference like the official SQuAD eval script.
        - ``speechrecognition``, ``translation``, ``summarization``: ``text_col`` +
          ``label_col`` (single reference text per row)

        ``text_col`` is not read by ``evaluate`` -- the pipeline's own first stage decides which
        column it consumes, so a mismatched ``text_col`` goes undetected.

        ``speechrecognition``'s word error rate is case-sensitive, matching `jiwer
        <https://github.com/jitsi/jiwer>`_'s default.

        ``summarization``'s ROUGE never stems (matches `rouge_score
        <https://github.com/google-research/google-research/tree/master/rouge>`_'s
        ``use_stemmer=False`` default) -- published ROUGE numbers for benchmarks like XSum/
        CNN-DailyMail typically use Porter stemming, so a real model's score here will read lower
        than those published figures on stem-sensitive text for reasons that have nothing to do
        with the model.

        ``predicted_col`` picks which of the pipeline's output columns to score. Optional: by
        default ``evaluate`` takes the last column of the type this task expects. Pass this when
        a pipeline has several stages sharing that type and the default picks the wrong one.
        Reported back on ``AccuracyReport.scored_columns``. For ``dependencyparsing`` this
        overrides only the ``dependency``-typed column; use ``predicted_labeled_dependency_col``
        for the ``labeled_dependency``-typed one.

        ``top_k`` sizes the top-k window for ``imageclassification``'s extra
        ``top{top_k}Accuracy`` metric; every other task ignores it.
        """
        task = task.lower()
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unknown benchmark task '{task}'. Supported: {sorted(SUPPORTED_TASKS)}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if label_col not in gold_data.columns:
            raise ValueError(f"gold_data must contain a '{label_col}' column")

        gold_input_fields = gold_data.schema.fields
        # Only cached for tasks that actually reuse it (see _TASKS_NEEDING_CACHE) -- caching the
        # whole annotated DataFrame for a single-pass task costs real memory for no benefit.
        predicted = pipeline_model.transform(gold_data)
        if task in _TASKS_NEEDING_CACHE:
            predicted = predicted.persist()

        resolved = []

        def resolve(annotator_type, override, param):
            name = _find_output_column(
                task, annotator_type, gold_input_fields, predicted, override, param)
            if name not in resolved:
                resolved.append(name)
            return name

        def find_col(annotator_type, override=predicted_col, param="predicted_col"):
            return resolve(annotator_type, override, param)

        def find_labeled_dependency_col():
            return resolve("labeled_dependency", predicted_labeled_dependency_col,
                           "predicted_labeled_dependency_col")

        try:
            report = Benchmark._evaluate_against(
                task, predicted, gold_data, label_col, find_col,
                find_labeled_dependency_col, top_k)
            report.scored_columns = resolved
            return report
        finally:
            predicted.unpersist()

    @staticmethod
    def _evaluate_against(task, predicted, gold_data, label_col, find_col,
                          find_labeled_dependency_col, top_k):

        if task == "ner":
            pred_col = find_col(_ANNOTATOR_TYPE["ner"])
            df = predicted.withColumn("__pred", F.expr(f"transform(`{pred_col}`, x -> x.result)"))
            rdd = df.select("__pred", label_col).rdd.map(lambda r: (r[0], r[1]))
            return _span_f1(task, rdd)

        if task == "wordsegmentation":
            pred_col = find_col(_ANNOTATOR_TYPE["wordsegmentation"])
            df = predicted.withColumn(
                "__predBoundaries", F.expr(f"transform(`{pred_col}`, x -> array(x.begin, x.end))"))
            rdd = df.select("__predBoundaries", label_col).rdd.map(
                lambda r: (
                    {(b[0], b[1]) for b in r[0]},
                    {_parse_int_pair(s) for s in r[1]},
                ))
            return _span_f1_boundaries(task, rdd)

        if task == "pos":
            pred_col = find_col(_ANNOTATOR_TYPE["pos"])
            df = predicted.withColumn("__pred", F.expr(f"transform(`{pred_col}`, x -> x.result)"))
            rdd = df.select("__pred", label_col).rdd.flatMap(
                lambda r: _zip_equal_length(task, r[0], r[1]))
            return _label_accuracy(task, rdd)

        if task in ("classification", "spellcheck", "languagedetection"):
            pred_col = find_col(_ANNOTATOR_TYPE[task])
            df = predicted.withColumn("__pred", F.expr(_single_result_expr(pred_col, _NO_PREDICTION)))
            rdd = df.select("__pred", label_col).rdd.map(lambda r: (r[0], r[1]))
            return _label_accuracy(task, rdd)

        if task == "imageclassification":
            pred_col = find_col(_ANNOTATOR_TYPE["imageclassification"])
            rdd = predicted.select(pred_col, label_col).rdd.map(
                lambda r: (_ranked_labels(r[0][0]["metadata"]) if r[0] else [], r[1]))
            return _label_accuracy_top_k(task, rdd, k=top_k)

        if task == "dependencyparsing":
            dep_col = find_col("dependency")
            labeled_dep_col = find_labeled_dependency_col()
            df = predicted \
                .withColumn("__head", F.expr(
                    f"transform(`{dep_col}`, x -> CAST(element_at(x.metadata, 'head') AS INT))")) \
                .withColumn("__label", F.expr(f"transform(`{labeled_dep_col}`, x -> x.result)"))
            rdd = df.select("__head", "__label", label_col).rdd.map(
                lambda r: (list(zip(r[0], r[1])), [_parse_head_label(s) for s in r[2]]))
            return _dependency_accuracy(task, rdd)

        metric = _TEXT_SIMILARITY_METRIC[task]
        annotator_type = "chunk" if metric == "squad" else "document"
        pred_col = find_col(annotator_type)
        df = predicted.withColumn("__pred", F.expr(_single_result_expr(pred_col, "")))

        if metric == "squad":
            multi_ref = dict(gold_data.dtypes)[label_col].startswith("array")
            rdd = df.select("__pred", label_col).rdd.map(
                lambda r: (r[0], list(r[1]) if multi_ref else ([] if r[1] is None else [r[1]])))
            overall, support = _squad_em_f1(rdd)
            return AccuracyReport(task, overall, {}, support)

        rdd = df.select("__pred", label_col).rdd.map(lambda r: (r[0], r[1]))
        return _text_similarity(task, rdd, metric)


def _parse_int_pair(s):
    a, b = s.split(":")
    return int(a), int(b)


def _parse_head_label(s):
    idx = s.index(":")
    return int(s[:idx]), s[idx + 1:]


_RESERVED_METADATA_KEYS = {
    "sentence", "image", "chunk", "score", "height", "width", "nChannels", "mode", "origin",
}


def _unwrap_label_key(key):
    # Some classifiers (e.g. ViTForImageClassification, see ViTClassifier.scala:184) build
    # metadata keys via `Option(...).toString` without ever unwrapping the Option first, so
    # EVERY class name key arrives as the literal string "Some(damselfly)" rather than
    # "damselfly", and a lookup miss arrives as bare "None". This is a bug in that annotator,
    # not a rare edge case -- confirmed against a real `image_classifier_vit_base_patch16_224`
    # prediction, where all 15 candidate keys were "Some(...)"-wrapped and only the miss case
    # was bare "None". Filtering "Some(...)" out entirely (rather than unwrapping it) discards
    # every real class on every row and silently reports 0% accuracy regardless of how good
    # the model is. Unwrap "Some(x)" to its inner label `x` instead -- still rejecting bare
    # "None" (a genuine no-match) and empty "Some()" -- while passing through any key that was
    # never wrapped, so classifiers whose metadata keys are already clean labels keep working
    # unchanged.
    if key == "None":
        return None
    if key.startswith("Some(") and key.endswith(")"):
        inner = key[5:-1]
        return inner if inner else None
    return key


def _ranked_labels(metadata):
    scored = []
    for k, v in metadata.items():
        if k in _RESERVED_METADATA_KEYS:
            continue
        label = _unwrap_label_key(k)
        if label is None:
            continue
        try:
            scored.append((label, float(v)))
        except (TypeError, ValueError):
            continue
    scored.sort(key=lambda kv: -kv[1])
    return [k for k, _ in scored]



def _label_accuracy(task, pairs):
    """`pairs` is an RDD of (predictedLabel, goldLabel); only the small set of distinct labels
    is ever collected to the driver, not the whole RDD."""
    from pyspark.mllib.evaluation import MulticlassMetrics

    labels = sorted(pairs.flatMap(lambda pg: pg).distinct().collect())
    if not labels:
        return AccuracyReport(task, {}, {}, 0)

    label_index = {label: i for i, label in enumerate(labels)}
    indexed_rdd = pairs.map(lambda pg: (float(label_index[pg[0]]), float(label_index[pg[1]])))
    indexed_rdd.persist()
    support = indexed_rdd.count()
    metrics = MulticlassMetrics(indexed_rdd)

    overall = {
        "accuracy": metrics.accuracy,
        "weightedPrecision": metrics.weightedPrecision,
        "weightedRecall": metrics.weightedRecall,
        "weightedF1": metrics.weightedFMeasure(),
    }
    # MulticlassMetrics raises for a label with no predictions or no gold occurrences.
    def safe_metric(fn, label):
        try:
            return fn(label)
        except Exception as e:
            if "NoSuchElementException" in str(e):
                return 0.0
            raise

    per_class = {}
    for label, idx in label_index.items():
        per_class[label] = {
            "precision": safe_metric(metrics.precision, float(idx)),
            "recall": safe_metric(metrics.recall, float(idx)),
            "f1": safe_metric(metrics.fMeasure, float(idx)),
        }
    indexed_rdd.unpersist()
    return AccuracyReport(task, overall, per_class, support)


def _label_accuracy_top_k(task, pairs, k):
    """`pairs` is an RDD of (topKPredictedLabels, goldLabel), best-first.

    Scores the pipeline's own top-1 label the same way :func:`_label_accuracy` does, and reports
    top-k as an additional ``top{k}Accuracy`` metric alongside it.
    """

    def score(top_k_gold):
        top_k, gold = top_k_gold
        top_1 = top_k[0] if top_k else _NO_PREDICTION
        return (top_1, gold, 1 if gold in top_k[:k] else 0)

    scored = pairs.map(score)
    scored.persist()
    try:
        hits, n = scored.map(lambda r: (r[2], 1)).fold(
            (0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1]))
        report = _label_accuracy(task, scored.map(lambda r: (r[0], r[1])))
        report.overall[f"top{k}Accuracy"] = (hits / n) if n else 0.0
        return report
    finally:
        scored.unpersist()



def _split_tag(tag):
    """"E"/"L" (IOBES "end"/BILOU "last") and "S"/"U" (IOBES "single"/BILOU "unit") are aliased
    onto a shared "E"/"S" bucket, so _extract_spans handles BIO/IOB2, IOBES, and BILOU uniformly."""
    if tag is None or tag == "O":
        return "O", None
    dash = tag.find("-")
    if dash < 0:
        return "B", tag
    prefix = {"L": "E", "U": "S"}.get(tag[:dash], tag[:dash])
    return prefix, tag[dash + 1:]


def _extract_spans(tags):
    spans = set()
    start = -1
    current_type = None

    def close_span(end_exclusive):
        nonlocal start, current_type
        if start >= 0:
            spans.add((start, end_exclusive, current_type))
        start = -1
        current_type = None

    for i, tag in enumerate(tags):
        prefix, t = _split_tag(tag)
        if prefix == "O":
            close_span(i)
        elif prefix == "B":
            close_span(i)
            start = i
            current_type = t
        elif prefix == "I":
            if start < 0 or current_type != t:
                close_span(i)
                start = i
                current_type = t
        elif prefix == "E":
            # Closes a chunk inclusive of this token -- extends an already-open same-type span
            # if there is one, otherwise (malformed input, a type change mid-span, or a genuine
            # IOBES/BILOU-style single-token close with no preceding B-/I-) closes whatever was
            # open first (so it isn't silently dropped) before opening a one-token span here.
            if start < 0 or current_type != t:
                close_span(i)
                start = i
                current_type = t
            close_span(i + 1)
        elif prefix == "S":
            # Always a standalone single-token entity, regardless of any span already open
            # (which gets closed first, unaffected by this one).
            close_span(i)
            spans.add((i, i + 1, t))
        else:
            close_span(i)
    close_span(len(tags))
    return spans


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _score_span_rdd(task, span_pairs):
    """`span_pairs` is an RDD of (predictedSpans, goldSpans) sets, each span a (start, end,
    type) tuple; reduced via a distributed map (per-row TP/FP/FN counts) then fold, never
    collecting the raw spans themselves to the driver."""

    def row_counts(pg):
        pred_spans, gold_spans = pg
        types = {t for _, _, t in pred_spans} | {t for _, _, t in gold_spans}
        counts = {}
        for t in types:
            pred_t = {s for s in pred_spans if s[2] == t}
            gold_t = {s for s in gold_spans if s[2] == t}
            tp = len(pred_t & gold_t)
            counts[t] = (tp, len(pred_t) - tp, len(gold_t) - tp)
        return counts

    def merge(a, b):
        merged = dict(a)
        for t, (tp, fp, fn) in b.items():
            ptp, pfp, pfn = merged.get(t, (0, 0, 0))
            merged[t] = (ptp + tp, pfp + fp, pfn + fn)
        return merged

    totals = span_pairs.map(row_counts).fold({}, merge)

    overall_tp = sum(v[0] for v in totals.values())
    overall_fp = sum(v[1] for v in totals.values())
    overall_fn = sum(v[2] for v in totals.values())
    support = sum(v[0] + v[2] for v in totals.values())

    per_class = {t: _prf(*counts) for t, counts in totals.items()}
    return AccuracyReport(task, _prf(overall_tp, overall_fp, overall_fn), per_class, support)


def _span_f1(task, rows):
    span_pairs = rows.map(lambda pg: (_extract_spans(pg[0]), _extract_spans(pg[1])))
    return _score_span_rdd(task, span_pairs)


def _span_f1_boundaries(task, boundary_pairs):
    span_pairs = boundary_pairs.map(
        lambda pg: (
            {(s, e, "segment") for s, e in pg[0]},
            {(s, e, "segment") for s, e in pg[1]},
        ))
    return _score_span_rdd(task, span_pairs)



def _dependency_accuracy(task, rows):
    """`rows` is an RDD of (predicted, gold) per sentence, each a sequence of (headIndex,
    dependencyLabel) per token, reduced via a distributed map then fold."""

    def row_stats(pg):
        predicted, gold = pg
        if len(predicted) != len(gold):
            raise ValueError(
                f"Benchmark.evaluate(task='{task}'): predicted and gold sequences have "
                f"different lengths ({len(predicted)} vs {len(gold)}) for one row. label_col "
                "must align one-to-one with the pipeline's own tokenization for this task.")
        correct_head = correct_head_and_label = total = 0
        for (p_head, p_label), (g_head, g_label) in zip(predicted, gold):
            total += 1
            if p_head == g_head:
                correct_head += 1
                if p_label == g_label:
                    correct_head_and_label += 1
        return (correct_head, correct_head_and_label, total)

    correct_head, correct_head_and_label, total = rows.map(row_stats).fold(
        (0, 0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    uas = correct_head / total if total else 0.0
    las = correct_head_and_label / total if total else 0.0
    return AccuracyReport(task, {"uas": uas, "las": las}, {}, total)



def _squad_normalize(text):
    import re
    import string
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _squad_em_f1(pairs):
    """`pairs` is an RDD of (predictedAnswer, [goldAnswer, ...]). Scores each prediction
    against every reference and keeps the best, matching the official SQuAD eval script's
    ``metric_max_over_ground_truths``."""

    def row_scores(pg):
        pred, golds = pg
        hyp = _squad_normalize(pred)
        hyp_tokens = hyp.split()
        hyp_counts = Counter(hyp_tokens)

        best_em = best_f1 = 0.0
        for gold in golds:
            ref = _squad_normalize(gold)
            em = 1.0 if hyp == ref else 0.0
            ref_tokens = ref.split()
            overlap = sum((hyp_counts & Counter(ref_tokens)).values())
            precision = overlap / len(hyp_tokens) if hyp_tokens else 0.0
            recall = overlap / len(ref_tokens) if ref_tokens else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            best_em = max(best_em, em)
            best_f1 = max(best_f1, f1)
        return (best_em, best_f1, 1)

    em_sum, f1_sum, n = pairs.map(row_scores).fold(
        (0.0, 0.0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    if n == 0:
        return {}, 0
    return {"exactMatch": em_sum / n, "f1": f1_sum / n}, n


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            curr[j] = (prev[j - 1] if a[i - 1] == b[j - 1]
                       else 1 + min(prev[j - 1], prev[j], curr[j - 1]))
        prev = curr
    return prev[len(b)]


def _wer(pairs):
    """Ports Scala `TextSimilarityEngine.wer` (Levenshtein distance on whitespace-split tokens)
    instead of calling `jiwer`, so both languages tokenize identically."""

    def row_stats(pg):
        pred, gold = pg
        hyp = pred.split()
        ref = gold.split()
        return (_levenshtein(hyp, ref), len(ref), 1)

    total_edits, total_words, n = pairs.map(row_stats).fold(
        (0, 0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
    score = total_edits / total_words if total_words else 0.0
    return {"wer": score}, n


_BLEU_PUNCT_13A = re.compile(r"([\x7B-\x7E\x5B-\x60\x20-\x26\x28-\x2B\x3A-\x40\x2F])")


def _bleu_tokenize(text):
    """Ports sacrebleu's default `13a` tokenizer, mirroring Scala `TextSimilarityEngine.
    bleuTokenize`."""
    line = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = (line.replace("&quot;", "\"").replace("&amp;", "&")
                    .replace("&lt;", "<").replace("&gt;", ">"))
    line = " " + line + " "
    line = _BLEU_PUNCT_13A.sub(r" \1 ", line)
    line = re.sub(r"([^0-9])([.,])", r"\1 \2 ", line)
    line = re.sub(r"([.,])([^0-9])", r" \1 \2", line)
    line = re.sub(r"([0-9])(-)", r"\1 \2 ", line)
    return line.split()


def _ngrams(tokens, n):
    return Counter(zip(*(tokens[i:] for i in range(n))))


def _clipped_overlap(hyp_ngrams, ref_ngrams):
    return sum(min(c, ref_ngrams.get(g, 0)) for g, c in hyp_ngrams.items())


def _bleu(pairs):
    """Ports Scala `TextSimilarityEngine.bleu` (corpus-level n-gram precision + NIST smoothing +
    brevity penalty) instead of calling `sacrebleu`, so this stays fully distributed."""

    def row_stats(pg):
        pred, gold = pg
        hyp = _bleu_tokenize(pred)
        ref = _bleu_tokenize(gold)
        matched = [0] * 4
        total = [0] * 4
        for n in range(1, 5):
            hyp_ngrams = _ngrams(hyp, n)
            matched[n - 1] = _clipped_overlap(hyp_ngrams, _ngrams(ref, n))
            total[n - 1] = max(0, len(hyp) - n + 1)
        return (matched, total, len(hyp), len(ref), 1)

    def merge(a, b):
        m1, t1, h1, r1, n1 = a
        m2, t2, h2, r2, n2 = b
        return ([x + y for x, y in zip(m1, m2)], [x + y for x, y in zip(t1, t2)],
                h1 + h2, r1 + r2, n1 + n2)

    matched, total, hyp_len, ref_len, n = pairs.map(row_stats).fold(
        ([0] * 4, [0] * 4, 0, 0, 0), merge)
    if n == 0:
        return {}, 0

    # Matches sacrebleu's own zero-overlap early-return (mjpost/sacrebleu#141): no smoothing,
    # every precision stays exactly 0.
    precisions = [0.0] * 4
    if any(matched):
        smooth = 1.0
        for i in range(4):
            if total[i] == 0:
                break
            if matched[i] == 0:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * total[i])
            else:
                precisions[i] = matched[i] / total[i]

    if hyp_len >= ref_len:
        brevity_penalty = 1.0
    elif hyp_len == 0:
        brevity_penalty = 0.0
    else:
        brevity_penalty = math.exp(1.0 - ref_len / hyp_len)

    # A 0.0 precision here is the same "no smoothing happened" case Scala's log(0)==-Infinity
    # zeroes the score for.
    bleu_score = (0.0 if any(p == 0.0 for p in precisions)
                  else brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0))

    return {
        "bleu": bleu_score,
        "precision1": precisions[0],
        "precision2": precisions[1],
        "precision3": precisions[2],
        "precision4": precisions[3],
        "brevityPenalty": brevity_penalty,
    }, n


def _rouge_tokenize(text):
    """Approximates rouge-score's default tokenizer, mirroring Scala
    `TextSimilarityEngine.rougeTokenize`."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def _prf_tuple(overlap, pred_len, gold_len):
    """Like `_prf`, but as a `(precision, recall, f1)` tuple -- ROUGE reports one per n-gram
    order."""
    d = _prf(overlap, pred_len - overlap, gold_len - overlap)
    return (d["precision"], d["recall"], d["f1"])


def _lcs_length(a, b):
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            curr[j] = (prev[j - 1] + 1 if a[i - 1] == b[j - 1]
                       else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[len(b)]


def _rouge(pairs):
    """Ports Scala `TextSimilarityEngine.rouge` (n-gram overlap + LCS) instead of calling
    `rouge_score`, so no package needs to exist on the executors."""
    keys = ("rouge1", "rouge2", "rougeL")

    def row_scores(pg):
        pred, gold = pg
        hyp = _rouge_tokenize(pred)
        ref = _rouge_tokenize(gold)

        def ngram_overlap(n):
            overlap = _clipped_overlap(_ngrams(hyp, n), _ngrams(ref, n))
            return _prf_tuple(overlap, max(0, len(hyp) - n + 1), max(0, len(ref) - n + 1))

        lcs = _lcs_length(hyp, ref)
        return (ngram_overlap(1), ngram_overlap(2), _prf_tuple(lcs, len(hyp), len(ref)), 1)

    def merge(a, b):
        add3 = lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])
        return (add3(a[0], b[0]), add3(a[1], b[1]), add3(a[2], b[2]), a[3] + b[3])

    zero = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0)
    r1_sum, r2_sum, rL_sum, n = pairs.map(row_scores).fold(zero, merge)
    if n == 0:
        return {}, 0

    overall = {}
    for key, (p, r, f) in zip(keys, (r1_sum, r2_sum, rL_sum)):
        overall[f"{key}_precision"] = p / n
        overall[f"{key}_recall"] = r / n
        overall[f"{key}_f1"] = f / n
    return overall, n


def _text_similarity(task, pairs, metric):
    if metric == "squad":
        # Single-reference convenience path: wrap each gold answer in a one-element list.
        overall, support = _squad_em_f1(pairs.map(lambda pg: (pg[0], [pg[1]])))
    elif metric == "wer":
        overall, support = _wer(pairs)
    elif metric == "bleu":
        overall, support = _bleu(pairs)
    elif metric == "rouge":
        overall, support = _rouge(pairs)
    else:
        raise ValueError(f"Unknown text-similarity metric '{metric}'")
    return AccuracyReport(task, overall, {}, support)
