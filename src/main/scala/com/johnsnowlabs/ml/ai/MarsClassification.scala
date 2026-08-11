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
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, Sentence, TokenPiece}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

/** One MARS phrase: a contiguous span of the answer text (character offsets) with a single
  * importance weight, as grouped by [[MarsClassification.tag]].
  *
  * `end` is '''inclusive''', following `TokenPiece.begin`/`end` and the `Annotation` convention
  * used throughout this codebase. Consumers joining these against half-open spans (e.g.
  * `UncertaintyMetrics.alignByCharSpan`, which reconciles them against the generating model's
  * exclusive-end token spans) have to account for that.
  */
case class MarsPhrase(begin: Int, end: Int, importance: Float)

/** Runs the MARS token-importance model (a standard `BertForTokenClassification` with 3 raw
  * output logits per token: `[:, 0:2]` a phrase-boundary class decided by argmax, `[:, 2]` a
  * per-token importance score decided by sigmoid - see
  * [[https://arxiv.org/abs/2402.11756 Bakman et al. 2024]]) over a `(question, answer)` pair and
  * groups the answer into importance-weighted phrases.
  *
  * Reimplements the reference `TruthTorchLM.truth_methods.mars.MARS.get_importance_vector_MARS`
  * phrase-grouping algorithm using this codebase's own wordpiece tokenizer, which already exposes
  * exact character offsets (`TokenPiece.begin`/`end`) and word-boundary flags
  * (`TokenPiece.isWordStart`) - so grouping doesn't need the original's `word_ids()`-based
  * bookkeeping, just a direct translation of the same break condition.
  *
  * @param onnxWrapper
  *   the loaded MARS ONNX model (`BertForTokenClassification` shape, `num_labels=3`)
  * @param vocabulary
  *   the wordpiece vocabulary (`vocab.txt`) the model was trained with
  * @param caseSensitive
  *   whether to lowercase before tokenizing (the public `duygunuryldz/MARS` checkpoint is
  *   `bert-base-uncased`, so this should normally be `false`)
  */
private[johnsnowlabs] class MarsClassification(
    val onnxWrapper: OnnxWrapper,
    vocabulary: Map[String, Int],
    caseSensitive: Boolean = false)
    extends Serializable {

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  private val clsId = vocabulary("[CLS]")
  private val sepId = vocabulary("[SEP]")

  /** Sentence-pair BERT encoding, mirroring `tokenizer.encode_plus([question], words,
    * is_split_into_words=True, ...)`: `[CLS] question_pieces [SEP] answer_pieces [SEP]`, with
    * `token_type_ids` 0 over the question span (+ first `[SEP]`) and 1 over the answer span (+
    * second `[SEP]`). Truncates the question and answer independently so the answer - the part
    * whose importance we actually need - isn't crowded out by a long question.
    *
    * @return
    *   (token ids, segment ids, answer wordpieces actually included after truncation, with their
    *   original character offsets into `answer`)
    */
  private def encode(
      question: String,
      answer: String,
      maxSeqLength: Int): (Array[Int], Array[Int], Array[TokenPiece]) = {
    // hasBeginEnd=false: tokenizing a whole raw string from scratch (like
    // BertClassification.tokenizeDocument), not a single pre-tokenized unit - with the default
    // hasBeginEnd=true every token in the string would be stamped with the *sentence's* begin
    // (here always 0) instead of its own position.
    val basicTokenizer = new BasicTokenizer(caseSensitive, hasBeginEnd = false)
    val encoder = new WordpieceEncoder(vocabulary)

    def wordpieces(text: String): Array[TokenPiece] = {
      val tokens: Array[IndexedToken] =
        basicTokenizer.tokenize(Sentence(text, 0, math.max(text.length - 1, 0), 0))
      tokens.flatMap(token => encoder.encode(token))
    }

    val questionPieces = wordpieces(question)
    val answerPieces = wordpieces(answer)

    val budget = math.max(maxSeqLength - 3, 0) // reserve [CLS], [SEP], [SEP]
    val questionBudget = math.min(questionPieces.length, budget / 2)
    val truncatedQuestion = questionPieces.take(questionBudget)
    val answerBudget = math.max(budget - truncatedQuestion.length, 0)
    val truncatedAnswer = answerPieces.take(answerBudget)

    val ids =
      Array(clsId) ++ truncatedQuestion.map(_.pieceId) ++ Array(sepId) ++
        truncatedAnswer.map(_.pieceId) ++ Array(sepId)
    val typeIds =
      Array.fill(truncatedQuestion.length + 2)(0) ++ Array.fill(truncatedAnswer.length + 1)(1)

    (ids, typeIds, truncatedAnswer)
  }

  private def getRawScoresWithOnnx(inputIds: Array[Int], typeIds: Array[Int]): Array[Float] = {
    val (runner, env) = onnxWrapper.getSession(onnxSessionOptions)

    val tokenTensor =
      OnnxTensor.createTensor(env, Array(inputIds.map(_.toLong)))
    val maskTensor =
      OnnxTensor.createTensor(env, Array(Array.fill(inputIds.length)(1L)))
    val segmentTensor =
      OnnxTensor.createTensor(env, Array(typeIds.map(_.toLong)))

    val inputs = Map(
      "input_ids" -> tokenTensor,
      "attention_mask" -> maskTensor,
      "token_type_ids" -> segmentTensor).asJava

    try {
      val results = runner.run(inputs)
      try {
        results.get("logits").get().asInstanceOf[OnnxTensor].getFloatBuffer.array()
      } finally if (results != null) results.close()
    } finally {
      tokenTensor.close()
      maskTensor.close()
      segmentTensor.close()
    }
  }

  /** Runs the model over `(question, answer)` and groups the answer into importance-weighted
    * phrases with character offsets into `answer`.
    *
    * @param maxSeqLength
    *   maximum total sequence length (question + answer + 3 special tokens); longer inputs are
    *   truncated, splitting the budget evenly between question and answer
    * @return
    *   phrases in answer order, covering every wordpiece of `answer` that fit within
    *   `maxSeqLength` (empty if the answer was truncated away entirely, e.g. an extremely long
    *   question)
    */
  def tag(question: String, answer: String, maxSeqLength: Int = 512): Array[MarsPhrase] = {
    val (inputIds, typeIds, answerPieces) = encode(question, answer, maxSeqLength)
    if (answerPieces.isEmpty) return Array.empty

    val rawScores = getRawScoresWithOnnx(inputIds, typeIds)
    require(
      rawScores.length == inputIds.length * MarsClassification.NumLabels,
      s"Unexpected MARS model output size: got ${rawScores.length} floats for " +
        s"${inputIds.length} tokens (expected ${inputIds.length * MarsClassification.NumLabels}, " +
        s"${MarsClassification.NumLabels} labels per token). Is this really a " +
        "MARS/BertForTokenClassification(num_labels=3) ONNX export?")
    val logitsPerToken: Array[Array[Float]] =
      rawScores.grouped(MarsClassification.NumLabels).toArray

    // Answer wordpieces start right after [CLS] + question pieces + [SEP].
    val answerStart = inputIds.length - answerPieces.length - 1 // -1 for the trailing [SEP]
    MarsClassification.groupPhrases(
      answerPieces,
      logitsPerToken.slice(answerStart, answerStart + answerPieces.length))
  }
}

private[johnsnowlabs] object MarsClassification {

  /** The MARS head's output width: `[0:2]` is a phrase-boundary class decided by argmax, `[2]` is
    * a per-token importance score decided by sigmoid.
    */
  val NumLabels: Int = 3

  /** Groups the answer's wordpieces into importance-weighted phrases, given one `NumLabels`-wide
    * logits row per answer wordpiece (already sliced out of the full sequence).
    *
    * A phrase break happens at a wordpiece that both starts a new word and is classified as a
    * boundary, so wordpiece continuations (`##...`) never start a phrase - a direct translation
    * of the reference implementation's break condition. Each phrase takes the importance of its
    * first token, and spans use inclusive `end` (see [[MarsPhrase]]).
    *
    * Kept free of ONNX and Spark types so the grouping can be tested against hand-built logits.
    */
  def groupPhrases(
      answerPieces: Array[TokenPiece],
      logitsPerPiece: Array[Array[Float]]): Array[MarsPhrase] = {
    require(
      answerPieces.length == logitsPerPiece.length,
      s"Expected one logits row per answer wordpiece, got ${logitsPerPiece.length} rows for " +
        s"${answerPieces.length} wordpieces")
    if (answerPieces.isEmpty) return Array.empty

    val classArgmax: Array[Int] =
      logitsPerPiece.map(logits => if (logits(1) > logits(0)) 1 else 0)
    val importanceSigmoid: Array[Float] =
      logitsPerPiece.map(logits => (1.0 / (1.0 + math.exp(-logits(2)))).toFloat)

    val phrases = ArrayBuffer[MarsPhrase]()
    var i = 0
    val n = answerPieces.length
    while (i < n) {
      var j = i + 1
      while (j < n && !(answerPieces(j).isWordStart && classArgmax(j) == 0)) {
        j += 1
      }
      phrases += MarsPhrase(answerPieces(i).begin, answerPieces(j - 1).end, importanceSigmoid(i))
      i = j
    }
    phrases.toArray
  }
}
