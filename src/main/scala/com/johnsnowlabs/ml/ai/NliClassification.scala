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
package com.johnsnowlabs.ml.ai

import ai.onnxruntime.OnnxTensor
import com.johnsnowlabs.ml.onnx.{OnnxSession, OnnxWrapper}
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/** Runs a BERT-style sequence classification model trained on NLI (natural language inference,
  * e.g. MNLI) to compute the entailment probability of a `(premise, hypothesis)` pair, for
  * [[com.johnsnowlabs.nlp.annotators.uncertainty.SampleEntailmentMatrix]]'s bidirectional
  * entailment clustering backend.
  *
  * Encoding and inference mirror [[MarsClassification]]'s (own verified) sentence-pair pattern:
  * `[CLS] premise_pieces [SEP] hypothesis_pieces [SEP]`, real `token_type_ids` (0 over the
  * premise span + first `[SEP]`, 1 over the hypothesis span + second `[SEP]`) - unlike
  * [[MarsClassification]] this model pools to a single `[CLS]` classification vector, so the ONNX
  * output is one `numLabels`-length logits row per input, not one per token.
  *
  * @param onnxWrapper
  *   the loaded NLI ONNX model (standard `BertForSequenceClassification` shape)
  * @param vocabulary
  *   the wordpiece vocabulary (`vocab.txt`) the model was trained with
  * @param labels
  *   label name -> output index, as trained (e.g. `{"contradiction"->0, "neutral"->1,
  *   "entailment"->2}` for a typical MNLI checkpoint - order is model-specific, resolved by name)
  * @param caseSensitive
  *   whether to lowercase before tokenizing
  */
private[johnsnowlabs] class NliClassification(
    val onnxWrapper: OnnxWrapper,
    vocabulary: Map[String, Int],
    labels: Map[String, Int],
    caseSensitive: Boolean = true)
    extends Serializable {

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  require(
    vocabulary.contains("[CLS]") && vocabulary.contains("[SEP]"),
    "The NLI model's vocab.txt must contain the [CLS] and [SEP] wordpieces this annotator uses " +
      "to build sentence-pair inputs.")

  // An uncased checkpoint's vocabulary has no capitalised entries at all, so running it
  // case-sensitively turns every proper noun into [UNK] - and then "the capital of France is
  // Paris" and "...is London" tokenize identically and score identically. That silently destroys
  // exactly the distinction this classifier exists to make, so it is worth shouting about.
  if (caseSensitive && !vocabulary.keysIterator.exists(_.exists(_.isUpper))) {
    NliClassification.logger.warn(
      "This NLI model's vocabulary contains no capitalised wordpieces, which means it is an " +
        "uncased checkpoint, but caseSensitive is true. Every capitalised token (notably proper " +
        "nouns - the answers this is usually asked to compare) will be encoded as [UNK], and " +
        "texts differing only in those tokens will receive identical entailment scores. Set " +
        "setCaseSensitive(false).")
  }

  private val entailmentIndex: Int = labels
    .find { case (name, _) => name.toLowerCase.startsWith("entail") }
    .map(_._2)
    .getOrElse(
      throw new IllegalArgumentException(
        "NLI model labels must include an entailment label (e.g. 'entailment'), one of: " +
          labels.keys.mkString(", ")))

  private val padId = vocabulary.getOrElse("[PAD]", 0)

  private def encode(
      premise: String,
      hypothesis: String,
      maxSeqLength: Int): (Array[Int], Array[Int]) =
    NliClassification.encodePair(vocabulary, caseSensitive, premise, hypothesis, maxSeqLength)

  /** Runs one padded batch of encoded pairs, returning one `numLabels`-wide logits row per pair.
    */
  private def getRawScoresWithOnnx(batch: Seq[(Array[Int], Array[Int])]): Array[Array[Float]] = {
    val (runner, env) = onnxWrapper.getSession(onnxSessionOptions)

    val maxLength = batch.map(_._1.length).max
    val paddedIds =
      batch.map { case (ids, _) =>
        ids.map(_.toLong) ++ Array.fill(maxLength - ids.length)(padId.toLong)
      }.toArray
    val masks =
      batch.map { case (ids, _) =>
        Array.fill(ids.length)(1L) ++ Array.fill(maxLength - ids.length)(0L)
      }.toArray
    val paddedTypes =
      batch.map { case (_, typeIds) =>
        typeIds.map(_.toLong) ++ Array.fill(maxLength - typeIds.length)(0L)
      }.toArray

    val tokenTensor = OnnxTensor.createTensor(env, paddedIds)
    val maskTensor = OnnxTensor.createTensor(env, masks)
    val segmentTensor = OnnxTensor.createTensor(env, paddedTypes)

    val inputs = Map(
      "input_ids" -> tokenTensor,
      "attention_mask" -> maskTensor,
      "token_type_ids" -> segmentTensor).asJava

    try {
      val results = runner.run(inputs)
      try {
        val flat = results.get("logits").get().asInstanceOf[OnnxTensor].getFloatBuffer.array()
        require(
          flat.length == batch.length * labels.size,
          s"Unexpected NLI model output size: got ${flat.length} floats for ${batch.length} " +
            s"pairs, expected ${batch.length * labels.size} (one per label per pair: " +
            s"${labels.keys.mkString(", ")}). Is this really a BertForSequenceClassification " +
            "NLI ONNX export?")
        flat.grouped(labels.size).toArray
      } finally if (results != null) results.close()
    } finally {
      tokenTensor.close()
      maskTensor.close()
      segmentTensor.close()
    }
  }

  private def softmax(logits: Array[Float]): Array[Float] = {
    val maxLogit = logits.max
    val exps = logits.map(l => math.exp((l - maxLogit).toDouble))
    val total = exps.sum
    exps.map(e => (e / total).toFloat)
  }

  /** Probability that `premise` entails `hypothesis`, per this NLI model. Directional: calling
    * with the arguments swapped is a different (generally different-valued) question.
    */
  def entailmentProbability(premise: String, hypothesis: String, maxSeqLength: Int = 512): Float =
    entailmentProbabilities(Seq((premise, hypothesis)), maxSeqLength, batchSize = 1).head

  /** Entailment probability for each `(premise, hypothesis)` pair, in input order.
    *
    * Pairs are padded and run `batchSize` at a time rather than one session call each: the caller
    * ([[com.johnsnowlabs.nlp.annotators.uncertainty.SampleEntailmentMatrix]]) needs every ordered
    * pair of a row's samples, which is `n * (n - 1)` calls, so per-pair session overhead
    * dominates quickly.
    */
  def entailmentProbabilities(
      pairs: Seq[(String, String)],
      maxSeqLength: Int = 512,
      batchSize: Int = 8): Array[Float] = {
    if (pairs.isEmpty) return Array.empty
    require(batchSize >= 1, "batchSize must be at least 1")
    pairs
      .map { case (premise, hypothesis) => encode(premise, hypothesis, maxSeqLength) }
      .grouped(batchSize)
      .flatMap(batch =>
        getRawScoresWithOnnx(batch).map(logits => softmax(logits)(entailmentIndex)))
      .toArray
  }
}

private[johnsnowlabs] object NliClassification {

  private val logger: Logger = LoggerFactory.getLogger("NliClassification")

  /** Sentence-pair BERT encoding: `[CLS] premise [SEP] hypothesis [SEP]`, with `token_type_ids` 0
    * over the premise span (+ first `[SEP]`) and 1 over the hypothesis span (+ second `[SEP]`).
    * Premise and hypothesis are truncated independently so a long premise cannot crowd the
    * hypothesis out entirely.
    *
    * Kept free of ONNX types so the encoding and truncation arithmetic can be tested against a
    * small hand-written vocabulary.
    *
    * @return
    *   (token ids, segment ids), always of equal length and never longer than `maxSeqLength`
    */
  def encodePair(
      vocabulary: Map[String, Int],
      caseSensitive: Boolean,
      premise: String,
      hypothesis: String,
      maxSeqLength: Int): (Array[Int], Array[Int]) = {
    // hasBeginEnd=false: tokenizing a whole raw string from scratch, not a single pre-tokenized
    // unit - see the MarsClassification.encode note on why the default (true) is wrong here.
    val basicTokenizer = new BasicTokenizer(caseSensitive, hasBeginEnd = false)
    val encoder = new WordpieceEncoder(vocabulary)

    def pieceIds(text: String): Array[Int] = {
      if (text.isEmpty) return Array.empty
      val tokens: Array[IndexedToken] =
        basicTokenizer.tokenize(Sentence(text, 0, text.length - 1, 0))
      tokens.flatMap(token => encoder.encode(token)).map(_.pieceId)
    }

    val premiseIds = pieceIds(premise)
    val hypothesisIds = pieceIds(hypothesis)

    val budget = math.max(maxSeqLength - 3, 0) // reserve [CLS], [SEP], [SEP]
    val premiseBudget = math.min(premiseIds.length, budget / 2)
    val truncatedPremise = premiseIds.take(premiseBudget)
    val hypothesisBudget = math.max(budget - truncatedPremise.length, 0)
    val truncatedHypothesis = hypothesisIds.take(hypothesisBudget)

    val ids =
      Array(vocabulary("[CLS]")) ++ truncatedPremise ++ Array(vocabulary("[SEP]")) ++
        truncatedHypothesis ++ Array(vocabulary("[SEP]"))
    val typeIds =
      Array.fill(truncatedPremise.length + 2)(0) ++ Array.fill(truncatedHypothesis.length + 1)(1)

    (ids, typeIds)
  }
}
