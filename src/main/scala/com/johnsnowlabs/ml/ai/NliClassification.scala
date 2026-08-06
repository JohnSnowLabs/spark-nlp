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

  private val clsId = vocabulary("[CLS]")
  private val sepId = vocabulary("[SEP]")

  private val entailmentIndex: Int = labels
    .find { case (name, _) => name.toLowerCase.startsWith("entail") }
    .map(_._2)
    .getOrElse(
      throw new IllegalArgumentException(
        "NLI model labels must include an entailment label (e.g. 'entailment'), one of: " +
          labels.keys.mkString(", ")))

  private def encode(
      premise: String,
      hypothesis: String,
      maxSeqLength: Int): (Array[Int], Array[Int]) = {
    // hasBeginEnd=false: tokenizing a whole raw string from scratch, not a single pre-tokenized
    // unit - see the MarsClassification.encode note on why the default (true) is wrong here.
    val basicTokenizer = new BasicTokenizer(caseSensitive, hasBeginEnd = false)
    val encoder = new WordpieceEncoder(vocabulary)

    def pieceIds(text: String): Array[Int] = {
      val tokens: Array[IndexedToken] =
        basicTokenizer.tokenize(Sentence(text, 0, math.max(text.length - 1, 0), 0))
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
      Array(clsId) ++ truncatedPremise ++ Array(sepId) ++ truncatedHypothesis ++ Array(sepId)
    val typeIds =
      Array.fill(truncatedPremise.length + 2)(0) ++ Array.fill(truncatedHypothesis.length + 1)(1)

    (ids, typeIds)
  }

  private def getRawScoresWithOnnx(inputIds: Array[Int], typeIds: Array[Int]): Array[Float] = {
    val (runner, env) = onnxWrapper.getSession(onnxSessionOptions)

    val tokenTensor = OnnxTensor.createTensor(env, Array(inputIds.map(_.toLong)))
    val maskTensor = OnnxTensor.createTensor(env, Array(Array.fill(inputIds.length)(1L)))
    val segmentTensor = OnnxTensor.createTensor(env, Array(typeIds.map(_.toLong)))

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

  private def softmax(logits: Array[Float]): Array[Float] = {
    val maxLogit = logits.max
    val exps = logits.map(l => math.exp((l - maxLogit).toDouble))
    val total = exps.sum
    exps.map(e => (e / total).toFloat)
  }

  /** Probability that `premise` entails `hypothesis`, per this NLI model. Directional: calling
    * with the arguments swapped is a different (generally different-valued) question.
    */
  def entailmentProbability(
      premise: String,
      hypothesis: String,
      maxSeqLength: Int = 512): Float = {
    val (ids, typeIds) = encode(premise, hypothesis, maxSeqLength)
    val logits = getRawScoresWithOnnx(ids, typeIds)
    require(
      logits.length == labels.size,
      s"Unexpected NLI model output size: got ${logits.length} floats, expected " +
        s"${labels.size} (one per label: ${labels.keys.mkString(", ")}). Is this really a " +
        "BertForSequenceClassification NLI ONNX export?")
    softmax(logits)(entailmentIndex)
  }
}
