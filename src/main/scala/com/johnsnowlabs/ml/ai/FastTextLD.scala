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
import com.johnsnowlabs.nlp.annotators.common.Sentence
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

import java.nio.charset.StandardCharsets
import java.nio.{FloatBuffer, LongBuffer}
import scala.collection.JavaConverters._
import scala.collection.mutable

/** Turns text into the input ids of a supervised fastText model, as fastText's
  * `Dictionary::getLine` does: tokens are split on fastText's whitespace, `</s>` ends the line,
  * label tokens (`__label__...`) are skipped, an in-vocabulary token adds its own id, and each
  * token adds its character n-grams. A newline is whitespace here; fastText would end the line at
  * it.
  *
  * @param vocabulary
  *   word to input-matrix row
  * @param nwords
  *   number of words in the model's dictionary, where the n-gram rows start
  * @param minn
  *   shortest character n-gram
  * @param maxn
  *   longest character n-gram; 0 turns n-grams off
  * @param bucket
  *   number of n-gram rows
  */
private[johnsnowlabs] class FastTextFeaturizer(
    vocabulary: Map[String, Int],
    nwords: Int,
    minn: Int,
    maxn: Int,
    bucket: Int)
    extends Serializable {

  import FastTextFeaturizer._

  def featurize(text: String): Array[Int] = {
    val ids = mutable.ArrayBuilder.make[Int]
    val it = tokens(wellFormed(text))
    var ended = false
    // as in fastText, a literal end-of-line token ends the line
    while (!ended && it.hasNext) {
      val token = it.next()
      if (token == Eos) ended = true
      else if (!token.startsWith(LabelPrefix)) {
        vocabulary.get(token).foreach(ids += _)
        addSubwords(token, ids)
      }
    }
    vocabulary.get(Eos).foreach(ids += _)
    ids.result()
  }

  /** Lone surrogates become '?', as the UTF-8 encoder writes them, so that word lookup and
    * n-grams see the same text.
    */
  private def wellFormed(text: String): String =
    if (text.exists(Character.isSurrogate))
      new String(text.getBytes(StandardCharsets.UTF_8), StandardCharsets.UTF_8)
    else text

  private def tokens(text: String): Iterator[String] = {
    val out = mutable.ArrayBuffer[String]()
    var start = -1
    var i = 0
    while (i < text.length) {
      if (isWhitespace(text.charAt(i))) {
        if (start >= 0) out += text.substring(start, i)
        start = -1
      } else if (start < 0) start = i
      i += 1
    }
    if (start >= 0) out += text.substring(start)
    out.iterator
  }

  private def addSubwords(token: String, ids: mutable.ArrayBuilder[Int]): Unit =
    if (maxn > 0) {
      val word = (Bow + token + Eow).getBytes(StandardCharsets.UTF_8)
      var i = 0
      while (i < word.length) {
        if ((word(i) & 0xc0) != 0x80) {
          var j = i
          var n = 1
          while (j < word.length && n <= maxn) {
            j += 1
            while (j < word.length && (word(j) & 0xc0) == 0x80) j += 1
            if (n >= minn && !(n == 1 && (i == 0 || j == word.length)))
              ids += nwords + Integer.remainderUnsigned(hashInt(word, i, j), bucket)
            n += 1
          }
        }
        i += 1
      }
    }
}

private[johnsnowlabs] object FastTextFeaturizer {

  val Eos = "</s>"
  private val LabelPrefix = "__label__"
  private val Bow = "<"
  private val Eow = ">"

  private def isWhitespace(c: Char): Boolean =
    c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\u000b' || c == '\f' || c == '\u0000'

  /** fastText's FNV-1a: each byte is sign-extended (`int8_t`) before the XOR. */
  def hash(bytes: Array[Byte]): Long = hashInt(bytes, 0, bytes.length) & 0xffffffffL

  private def hashInt(bytes: Array[Byte], from: Int, until: Int): Int = {
    var h = 0x811c9dc5
    var k = from
    while (k < until) {
      h = (h ^ bytes(k)) * 16777619
      k += 1
    }
    h
  }
}

/** Language identification with a supervised fastText model (such as GlotLID) exported to ONNX.
  * The graph takes `input_ids` and `mask` of shape `[batch, features]` and returns the softmax
  * `probs` of shape `[batch, labels]`.
  *
  * @param onnxWrapper
  *   the exported model
  * @param featurizer
  *   fastText featurizer for the model's dictionary
  * @param labels
  *   labels in output order
  */
private[johnsnowlabs] class FastTextLD(
    val onnxWrapper: OnnxWrapper,
    featurizer: FastTextFeaturizer,
    labels: Array[String])
    extends Serializable {

  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  def predict(
      documents: Seq[Sentence],
      threshold: Float,
      thresholdLabel: String,
      coalesceSentences: Boolean,
      scriptPooling: Boolean,
      minLetters: Int,
      topK: Int): Seq[Annotation] = {

    val probs = scores(documents.map(_.content))

    def annotation(p: Array[Float], text: String, begin: Int, end: Int, index: Int) = {
      val (result, confidence) =
        FastTextLD.decide(p, labels, text, threshold, thresholdLabel, scriptPooling, minLetters)
      val scores = FastTextLD.topScores(p, labels, topK).map { case (l, s) => l -> s.toString }
      Annotation(
        annotatorType = AnnotatorType.LANGUAGE,
        begin = begin,
        end = end,
        result = result,
        metadata =
          Map("sentence" -> index.toString, "confidence" -> confidence.toString) ++ scores)
    }

    if (coalesceSentences) {
      val average = probs.transpose.map(column => column.sum / probs.length)
      Seq(
        annotation(
          average,
          documents.map(_.content).mkString(" "),
          documents.head.start,
          documents.last.end,
          documents.head.index))
    } else
      documents.zip(probs).map { case (sentence, p) =>
        annotation(p, sentence.content, sentence.start, sentence.end, sentence.index)
      }
  }

  /** Probabilities per text, in label order. Texts go to the model in length-sorted batches of at
    * most `maxCells` padded ids (a longer text goes alone), so one long sentence does not pad the
    * others of its document to its length.
    */
  def scores(texts: Seq[String], maxCells: Int = FastTextLD.MaxCells): Array[Array[Float]] = {
    val features = texts.map(featurizer.featurize).toArray
    val order = features.indices.sortBy(i => features(i).length)
    val out = new Array[Array[Float]](features.length)
    var start = 0
    while (start < order.length) {
      var end = start + 1
      while (end < order.length &&
        FastTextLD.fitsBatch(end - start + 1, features(order(end)).length, maxCells)) end += 1
      val batch = order.slice(start, end)
      run(batch.map(i => features(i))).zip(batch).foreach { case (p, i) => out(i) = p }
      start = end
    }
    out
  }

  private def run(rows: Seq[Array[Int]]): Array[Array[Float]] = {
    val width = math.max(1, rows.map(_.length).max)
    val ids = LongBuffer.allocate(rows.length * width)
    val mask = FloatBuffer.allocate(rows.length * width)
    rows.foreach { row =>
      var i = 0
      while (i < width) {
        if (i < row.length) {
          ids.put(row(i).toLong)
          mask.put(1f)
        } else {
          ids.put(0L)
          mask.put(0f)
        }
        i += 1
      }
    }
    ids.rewind()
    mask.rewind()

    val (session, env) = onnxWrapper.getSession(onnxSessionOptions)
    val shape = Array(rows.length.toLong, width.toLong)
    val idsTensor = OnnxTensor.createTensor(env, ids, shape)
    try {
      val maskTensor = OnnxTensor.createTensor(env, mask, shape)
      try {
        val results = session.run(Map("input_ids" -> idsTensor, "mask" -> maskTensor).asJava)
        try {
          val output = results.get("probs")
          require(
            output.isPresent,
            "model.onnx has no 'probs' output; export it again with the LanguageDetectorDL import notebook")
          val probs = output.get().asInstanceOf[OnnxTensor].getFloatBuffer
          val flat = new Array[Float](probs.remaining())
          probs.get(flat)
          require(
            flat.length == rows.length * labels.length,
            s"the model gives ${flat.length / rows.length} scores per text, " +
              s"but language.txt has ${labels.length} labels")
          // grouped would box every element
          Array.tabulate(rows.length)(r =>
            java.util.Arrays.copyOfRange(flat, r * labels.length, (r + 1) * labels.length))
        } finally results.close()
      } finally maskTensor.close()
    } finally idsTensor.close()
  }
}

private[johnsnowlabs] object FastTextLD {

  val MaxCells: Int = 1 << 16

  /** Whether `rows` texts padded to `width` ids stay within `maxCells`. */
  def fitsBatch(rows: Int, width: Int, maxCells: Int): Boolean =
    rows.toLong * math.max(1, width) <= maxCells

  private val Latin = "Latn"
  private val NonLanguagePrefix = "zxx_"

  /** The label for one text and the score it was judged on.
    *
    * With `scriptPooling`, a non-Latin top label is judged on the summed scores of every label in
    * its script, leaving out the non-language (`zxx_`) labels. With `minLetters` > 0, a
    * Latin-script top label (`_Latn`) on text with fewer letters gives `thresholdLabel`.
    */
  def decide(
      probs: Array[Float],
      labels: Array[String],
      text: String,
      threshold: Float,
      thresholdLabel: String,
      scriptPooling: Boolean,
      minLetters: Int): (String, Float) = {
    var best = 0
    var i = 1
    while (i < probs.length) {
      if (probs(i) > probs(best)) best = i
      i += 1
    }
    val label = labels(best)
    val confidence =
      if (scriptPooling) pooledScore(probs, labels, label).getOrElse(probs(best))
      else probs(best)
    val tooShort =
      minLetters > 0 && script(label).contains(Latin) && letterCount(text) < minLetters
    (if (confidence >= threshold && !tooShort) label else thresholdLabel, confidence)
  }

  /** The k best labels, highest first and the lower index first on ties; all of them for k = 0.
    */
  def topScores(probs: Array[Float], labels: Array[String], k: Int): Seq[(String, Float)] =
    if (k <= 0 || k >= probs.length)
      probs.indices.sortBy(i => -probs(i)).map(i => (labels(i), probs(i)))
    else {
      // partial insertion sort: keeps k labels without sorting all of them
      val best = new Array[Int](k)
      var size = 0
      var i = 0
      while (i < probs.length) {
        val p = probs(i)
        if (size < k || p > probs(best(size - 1))) {
          var j = if (size < k) size else k - 1
          while (j > 0 && probs(best(j - 1)) < p) {
            best(j) = best(j - 1)
            j -= 1
          }
          best(j) = i
          if (size < k) size += 1
        }
        i += 1
      }
      best.take(size).toSeq.map(i => (labels(i), probs(i)))
    }

  private def script(label: String): Option[String] = {
    val i = label.lastIndexOf('_')
    if (i > 0 && i < label.length - 1) Some(label.substring(i + 1)) else None
  }

  private def pooledScore(
      probs: Array[Float],
      labels: Array[String],
      top: String): Option[Float] =
    script(top).filter(s => s != Latin && !top.startsWith(NonLanguagePrefix)).map { s =>
      var sum = 0.0
      var i = 0
      while (i < labels.length) {
        if (!labels(i).startsWith(NonLanguagePrefix) && script(labels(i)).contains(s))
          sum += probs(i)
        i += 1
      }
      sum.toFloat
    }

  private def letterCount(text: String): Int = {
    var letters = 0
    var i = 0
    while (i < text.length) {
      val cp = text.codePointAt(i)
      if (Character.isLetter(cp)) letters += 1
      i += Character.charCount(cp)
    }
    letters
  }
}
