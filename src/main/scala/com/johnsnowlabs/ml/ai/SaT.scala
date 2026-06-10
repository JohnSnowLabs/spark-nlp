/*
 * Copyright 2017-2024 John Snow Labs
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
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper

import scala.collection.JavaConverters._

// ═══════════════════════════════════════════════════════════════════════════════
//  Data types used by the SaT inference pipeline
// ═══════════════════════════════════════════════════════════════════════════════

/** Result of tokenising a document with the XLM-R SentencePiece model.
  *
  * @param inputIds
  *   one XLM-R token id per sub-word piece (special tokens CLS/SEP are NOT included here – they
  *   are added per window later)
  * @param offsets
  *   for every token, its `(charStart inclusive, charEnd exclusive)` span in the original text;
  *   `offsets.length == inputIds.length`
  */
case class TokenizedText(inputIds: Array[Int], offsets: Array[(Int, Int)])

/** One overlapping slice of the document that is fed to the model in a single forward pass.
  *
  * @param tokenStart
  *   index (into the document-level token array) of the first real token in this window
  * @param tokenEnd
  *   index just past the last real token in this window (exclusive)
  * @param inputIds
  *   the token ids actually sent to ONNX: `CLS + realTokens + SEP`
  * @param attentionMask
  *   `1.0f` for every position in `inputIds` (padding to the batch width is added at run time)
  */
case class SaTWindow(
    tokenStart: Int,
    tokenEnd: Int,
    inputIds: Array[Long],
    attentionMask: Array[Float])

/** A detected sentence, expressed in the original document's character space.
  *
  * @param begin
  *   first character index (inclusive)
  * @param end
  *   last character index (inclusive – Spark NLP's annotation convention)
  * @param text
  *   the substring of the document this span covers
  */
case class SentenceSpan(begin: Int, end: Int, text: String)

/** One piece parsed out of a SentencePiece serialized proto.
  *
  * @param id
  *   the raw SentencePiece id (before the XLM-R `+1` shift)
  * @param begin
  *   byte offset (inclusive) of the piece in the UTF-8 encoding of the original text
  * @param end
  *   byte offset (exclusive) of the piece in the UTF-8 encoding of the original text
  */
case class SppPiece(id: Int, begin: Int, end: Int)

// ═══════════════════════════════════════════════════════════════════════════════
//  Inference engine
// ═══════════════════════════════════════════════════════════════════════════════

/** Under-the-hood inference engine for the SaT (Segment-any-Text / wtpsplit) sentence boundary
  * model.
  *
  * This class owns the whole pipeline that turns raw text into sentence spans:
  *
  *   1. '''Tokenise''' the text with the XLM-R SentencePiece model, keeping the exact character
  *      offset of every sub-word token ([[encodeWithOffsets]]).
  *   1. '''Window''' the token stream into overlapping chunks of at most 510 real tokens so any
  *      length of document fits the model's 512-position limit ([[makeWindows]]).
  *   1. '''Run ONNX''' on batches of windows; the model emits one boundary logit per token
  *      ([[runWindowedInference]]).
  *   1. '''Merge''' the overlapping windows back into a single logit per token using a weighted
  *      average ([[runWindowedInference]] + [[windowWeights]]).
  *   1. '''Decode''': sigmoid the logits, push each token's probability onto the last character
  *      of its span ([[tokenLogitsToCharProbs]]), then cut the text wherever the probability
  *      crosses the threshold ([[charProbsToSentenceSpans]]).
  *
  * It mirrors the design of the other `ml.ai` engines (e.g. [[XlmRoberta]]): it is constructed
  * with the raw [[OnnxWrapper]] and [[SentencePieceWrapper]], resolves the ONNX session options
  * once on the driver, and is broadcast to the executors by the annotator.
  *
  * @param onnxWrapper
  *   loaded SaT ONNX model (`model.onnx`)
  * @param spp
  *   loaded XLM-R SentencePiece model (`sentencepiece.bpe.model`)
  */
private[johnsnowlabs] class SaT(val onnxWrapper: OnnxWrapper, val spp: SentencePieceWrapper)
    extends Serializable {

  /** XLM-R special-token ids. `<s>` (CLS) = 0, `<pad>` = 1, `</s>` (SEP) = 2. */
  private val ClsId = 0
  private val SepId = 2

  /** HuggingFace's XLM-R tokenizer shifts every raw SentencePiece id by `+1` (`<s>=0, <pad>=1,
    * </s>=2, <unk>=3`, then real pieces start at `sp_id + 1`). Raw SentencePiece ids must
    * therefore be offset before being fed to the ONNX graph, exactly like
    * [[com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencepieceEncoder]] does with
    * `pieceIdOffset = 1`.
    */
  private val FairseqOffset = 1

  /** ONNX session options, resolved once on the driver at construction time (same pattern as
    * [[XlmRoberta]]). The resulting plain `Map[String, String]` is serializable and is carried
    * inside the broadcast without touching the native ORT runtime on executor threads.
    */
  private val onnxSessionOptions: Map[String, String] = new OnnxSession().getSessionOptions

  // ── Public API ────────────────────────────────────────────────────────────────

  /** Split a single document into sentence spans.
    *
    * @param text
    *   document text of any length
    * @param threshold
    *   a boundary is placed after a character once its probability is `>= threshold`
    * @param blockSize
    *   number of real tokens per ONNX window (must be `<= 510` for XLM-R)
    * @param stride
    *   how many tokens to advance between consecutive windows (smaller = more overlap)
    * @param batchSize
    *   how many windows to send to ONNX in a single forward pass
    * @param weighting
    *   `"hat"` (edge tokens trusted less) or `"uniform"` overlap weighting
    * @param trimWhitespace
    *   strip leading/trailing whitespace from each detected sentence
    * @return
    *   ordered, non-overlapping list of [[SentenceSpan]]s covering the whole document
    */
  def split(
      text: String,
      threshold: Float,
      blockSize: Int,
      stride: Int,
      batchSize: Int,
      weighting: String,
      trimWhitespace: Boolean): Seq[SentenceSpan] = {

    require(blockSize >= 1 && blockSize <= 510, "blockSize must be in [1, 510] for XLM-R/SaT.")

    if (text == null || text.isEmpty) return Seq.empty

    // 1. Tokenise with character offsets.
    val tokenized = encodeWithOffsets(text)

    // Degenerate input (e.g. only whitespace the tokenizer dropped): return the text as-is.
    if (tokenized.inputIds.isEmpty)
      return Seq(SentenceSpan(0, text.length - 1, text))

    // 2. Build overlapping windows of <= 510 real tokens each.
    val windows = makeWindows(tokenized.inputIds, blockSize, stride)

    // 3. Run ONNX on the windows and merge their overlapping logits into one logit per token.
    val mergedLogits =
      runWindowedInference(windows, tokenized.inputIds.length, batchSize, weighting)

    // 4. Project the per-token logits onto per-character boundary probabilities.
    val charProbs = tokenLogitsToCharProbs(text, mergedLogits, tokenized.offsets)

    // 5. Cut the text wherever the probability crosses the threshold.
    charProbsToSentenceSpans(text, charProbs, threshold, trimWhitespace)
  }

  // ── 1. Tokenisation (encoding) ──────────────────────────────────────────────────

  /** Encode `text` into XLM-R token ids together with the exact character span of every token.
    *
    * Both the ids and the offsets come from the SentencePiece *serialized proto*
    * (`encodeAsSerializedProto`), which records the original-text byte span (`begin`/`end`) of
    * each piece – the same information HuggingFace's fast tokenizer exposes via
    * `return_offsets_mapping=True`. This is robust to SentencePiece normalisation and arbitrary
    * whitespace, unlike walking the piece surfaces by hand.
    *
    *   1. raw SentencePiece ids are shifted by [[FairseqOffset]] to match the XLM-R vocabulary,
    *      and
    *   1. the proto's UTF-8 byte offsets are mapped back to Java (UTF-16) character indices.
    *
    * Special tokens (CLS/SEP) are NOT added here – [[makeWindows]] adds them per window.
    */
  private def encodeWithOffsets(text: String): TokenizedText = {
    val proto = spp.getSppModel.encodeAsSerializedProto(text)
    if (proto == null || proto.isEmpty) return TokenizedText(Array.empty, Array.empty)

    val pieces = parseSppPieces(proto)
    if (pieces.isEmpty) return TokenizedText(Array.empty, Array.empty)

    val byteToChar = buildByteToCharMap(text)
    val numBytes = byteToChar.length - 1

    val ids = new Array[Int](pieces.length)
    val offsets = new Array[(Int, Int)](pieces.length)
    for (i <- pieces.indices) {
      val SppPiece(rawId, beginByte, endByte) = pieces(i)
      ids(i) = rawId + FairseqOffset
      val rawStart = if (beginByte >= 0 && beginByte <= numBytes) byteToChar(beginByte) else 0
      val end = if (endByte >= 0 && endByte <= numBytes) byteToChar(endByte) else rawStart

      // SentencePiece's `begin` includes the leading whitespace that the `▁` metaspace stands for,
      // whereas HuggingFace's offset_mapping points at the first real character. Skip the leading
      // whitespace to match HF. (This only moves `start`; SaT decodes off `end`, so behaviour is
      // unchanged either way.) If the token is whitespace-only, keep the raw span.
      var start = rawStart
      while (start < end && text.charAt(start).isWhitespace) start += 1
      if (start >= end) start = rawStart

      offsets(i) = (start, end)
    }

    TokenizedText(ids, offsets)
  }

  /** Parse a serialized `SentencePieceText` protobuf into the pieces we need.
    *
    * Only the relevant fields are read; everything else is skipped. The (stable) wire layout is:
    *   - `SentencePieceText`: `pieces` = field 2 (repeated, length-delimited sub-message)
    *   - `SentencePiece`: `id` = field 2 (varint), `begin` = field 4 (varint), `end` = field 5
    *     (varint), where `begin`/`end` are byte offsets into the original input text.
    *
    * Reading the wire format directly avoids depending on a generated protobuf class.
    */
  private def parseSppPieces(buf: Array[Byte]): Array[SppPiece] = {
    val pieces = scala.collection.mutable.ArrayBuffer.empty[SppPiece]
    var pos = 0
    val len = buf.length
    while (pos < len) {
      val (tag, p1) = readVarint(buf, pos)
      pos = p1
      val field = (tag >> 3).toInt
      val wire = (tag & 0x7).toInt
      wire match {
        case 2 => // length-delimited
          val (l, p2) = readVarint(buf, pos)
          pos = p2
          val msgEnd = pos + l.toInt
          if (field == 2) pieces += parseOnePiece(buf, pos, msgEnd)
          pos = msgEnd
        case 0 => pos = readVarint(buf, pos)._2 // varint
        case 1 => pos += 8 // 64-bit
        case 5 => pos += 4 // 32-bit
        case _ => pos = len // unknown wire type – stop defensively
      }
    }
    pieces.toArray
  }

  /** Parse a single `SentencePiece` sub-message located in `buf[start, end)`. */
  private def parseOnePiece(buf: Array[Byte], start: Int, end: Int): SppPiece = {
    var pos = start
    var id = 0
    var begin = 0
    var fin = 0
    while (pos < end) {
      val (tag, p1) = readVarint(buf, pos)
      pos = p1
      val field = (tag >> 3).toInt
      val wire = (tag & 0x7).toInt
      wire match {
        case 0 =>
          val (v, p2) = readVarint(buf, pos)
          pos = p2
          field match {
            case 2 => id = v.toInt
            case 4 => begin = v.toInt
            case 5 => fin = v.toInt
            case _ => // ignore other varint fields
          }
        case 2 =>
          val (l, p2) = readVarint(buf, pos)
          pos = p2 + l.toInt // skip string fields (piece / surface)
        case 1 => pos += 8
        case 5 => pos += 4
        case _ => pos = end
      }
    }
    SppPiece(id, begin, fin)
  }

  /** Read a base-128 varint starting at `startPos`; returns `(value, nextPos)`. */
  private def readVarint(buf: Array[Byte], startPos: Int): (Long, Int) = {
    var result = 0L
    var shift = 0
    var pos = startPos
    var continue = true
    while (continue) {
      val b = buf(pos) & 0xff
      result |= (b.toLong & 0x7fL) << shift
      pos += 1
      if ((b & 0x80) == 0) continue = false else shift += 7
    }
    (result, pos)
  }

  /** Build a lookup from UTF-8 byte offset to Java (UTF-16) character index.
    *
    * `result(b)` is the character index of the code point whose UTF-8 encoding starts at byte
    * `b`; the trailing sentinel `result(numBytes) == text.length` lets an exclusive `end` byte
    * offset resolve cleanly. SentencePiece offsets always fall on code-point boundaries, so only
    * those byte positions are ever looked up.
    */
  private def buildByteToCharMap(text: String): Array[Int] = {
    val map = scala.collection.mutable.ArrayBuffer.empty[Int]
    var charPos = 0
    var i = 0
    val n = text.length
    while (i < n) {
      val cp = text.codePointAt(i)
      val charCount = Character.charCount(cp) // 1 or 2 UTF-16 units
      val byteCount = utf8Length(cp)
      var b = 0
      while (b < byteCount) { map += charPos; b += 1 }
      charPos += charCount
      i += charCount
    }
    map += charPos // sentinel: byte offset == numBytes -> text.length
    map.toArray
  }

  /** Number of bytes a single code point occupies in UTF-8. */
  private def utf8Length(cp: Int): Int =
    if (cp < 0x80) 1 else if (cp < 0x800) 2 else if (cp < 0x10000) 3 else 4

  // ── 2. Windowing ───────────────────────────────────────────────────────────────

  /** Slice the flat token array into overlapping windows of at most `blockSize` real tokens.
    *
    * Each window's `inputIds` are `CLS + tokenIds[tokenStart, tokenEnd) + SEP`, so the model
    * never sees more than `blockSize + 2 <= 512` positions at once. Consecutive windows start
    * `stride` tokens apart, so neighbouring windows overlap by `blockSize - stride` tokens; those
    * shared tokens get a prediction from more than one window and are merged later.
    *
    * @return
    *   a non-empty sequence of windows that together cover every token at least once
    */
  private def makeWindows(tokenIds: Array[Int], blockSize: Int, stride: Int): Seq[SaTWindow] = {
    val numTokens = tokenIds.length
    val windows = scala.collection.mutable.ArrayBuffer.empty[SaTWindow]

    var start = 0
    var continue = true
    while (continue) {
      val end = math.min(start + blockSize, numTokens)

      val realTokens = tokenIds.slice(start, end).map(_.toLong)
      val inputIds = Array(ClsId.toLong) ++ realTokens ++ Array(SepId.toLong)
      val mask = Array.fill(inputIds.length)(1.0f)

      windows += SaTWindow(
        tokenStart = start,
        tokenEnd = end,
        inputIds = inputIds,
        attentionMask = mask)

      if (end >= numTokens) continue = false // reached the end of the document
      else start += stride
    }

    windows.toSeq
  }

  // ── 3 + 4. ONNX forward pass and overlap merging ────────────────────────────────

  /** Run the model over all windows (in batches) and merge their overlapping per-token logits.
    *
    * For every window the model returns one logit per position; positions `1 .. realCount` are
    * the real tokens (position 0 is CLS, the last is SEP). Each real token's logit is added into
    * a document-level accumulator, weighted by [[windowWeights]], and finally divided by the
    * total weight so that tokens seen by several windows get a proper weighted average.
    *
    * @return
    *   one merged logit per document token (`length == numTokens`)
    */
  private def runWindowedInference(
      windows: Seq[SaTWindow],
      numTokens: Int,
      batchSize: Int,
      weighting: String): Array[Float] = {

    val mergedLogits = Array.fill(numTokens)(0.0f)
    val weightsSum = Array.fill(numTokens)(0.0f)
    val epsilon = 1e-8f

    val (runner, env) = onnxWrapper.getSession(onnxSessionOptions)

    for (batch <- windows.grouped(batchSize)) {
      // Pad every window in the batch to the longest one (pad id 0, mask 0.0f).
      val maxLen = batch.map(_.inputIds.length).max
      val flatIds = new Array[Long](batch.size * maxLen)
      val flatMask = new Array[Float](batch.size * maxLen)
      for ((window, bi) <- batch.zipWithIndex) {
        val wLen = window.inputIds.length
        System.arraycopy(window.inputIds, 0, flatIds, bi * maxLen, wLen)
        System.arraycopy(window.attentionMask, 0, flatMask, bi * maxLen, wLen)
      }

      val inputIdsTensor = OnnxTensor.createTensor(env, to2DLong(flatIds, batch.size, maxLen))
      val attentionMaskTensor =
        OnnxTensor.createTensor(env, to2DFloat(flatMask, batch.size, maxLen))
      val inputs =
        Map("input_ids" -> inputIdsTensor, "attention_mask" -> attentionMaskTensor).asJava

      try {
        val results = runner.run(inputs)
        try {
          // logits are flattened [batch, maxLen, 1] -> batch * maxLen floats.
          val rawLogits =
            results.get("logits").get().asInstanceOf[OnnxTensor].getFloatBuffer.array()

          for ((window, bi) <- batch.zipWithIndex) {
            val realCount = window.tokenEnd - window.tokenStart
            val weights = windowWeights(realCount, weighting)
            for (localIdx <- 0 until realCount) {
              val globalIdx = window.tokenStart + localIdx
              val logitIdx = bi * maxLen + (localIdx + 1) // +1 skips the CLS position
              if (logitIdx < rawLogits.length) {
                val w = weights(localIdx)
                mergedLogits(globalIdx) += rawLogits(logitIdx) * w
                weightsSum(globalIdx) += w
              }
            }
          }
        } finally {
          if (results != null) results.close()
        }
      } finally {
        inputIdsTensor.close()
        attentionMaskTensor.close()
      }
    }

    for (i <- mergedLogits.indices) mergedLogits(i) /= math.max(weightsSum(i), epsilon)
    mergedLogits
  }

  /** Per-token overlap weights for one window of `n` real tokens.
    *
    *   - `"hat"`: a triangular weighting that trusts tokens near the centre of the window more
    *     than tokens at the edges (edge tokens have less left/right context, so they are less
    *     reliable).
    *   - anything else (`"uniform"`): every token weighted equally.
    */
  private def windowWeights(n: Int, weighting: String): Array[Float] =
    if (weighting == "hat") {
      if (n <= 1) Array.fill(n)(1.0f)
      else {
        val center = (n - 1).toFloat / 2.0f
        Array.tabulate(n) { i =>
          val dist = math.abs(i - center) / math.max(center, 1.0f)
          math.max(1.0f - dist, 1e-3f) // never fully zero, so edge tokens still count a little
        }
      }
    } else Array.fill(n)(1.0f)

  // ── 5. Decoding (logits -> sentence spans) ──────────────────────────────────────

  /** Numerically-stable logistic sigmoid, computed in double precision. */
  private def sigmoid(x: Float): Float = (1.0 / (1.0 + math.exp(-x.toDouble))).toFloat

  /** Turn one logit per token into one boundary probability per character.
    *
    * Following the SaT convention, each token's probability is assigned to the '''last
    * character''' of its span (`offset.end - 1`). If two tokens happen to end on the same
    * character the larger probability wins. Characters not touched by any token keep probability
    * `0`.
    */
  private def tokenLogitsToCharProbs(
      text: String,
      logits: Array[Float],
      offsets: Array[(Int, Int)]): Array[Float] = {

    val charProbs = Array.fill(text.length)(0.0f)
    for (i <- logits.indices) {
      val (start, end) = offsets(i)
      if (end > start && end <= text.length) {
        val charIdx = end - 1
        val prob = sigmoid(logits(i))
        if (prob > charProbs(charIdx)) charProbs(charIdx) = prob
      }
    }
    charProbs
  }

  /** Cut the document into sentences wherever a character's boundary probability reaches the
    * threshold. A boundary after character `i` ends the current sentence at `i` (inclusive) and
    * starts the next sentence at `i + 1`. Any text after the final boundary becomes the last
    * sentence.
    */
  private def charProbsToSentenceSpans(
      text: String,
      charProbs: Array[Float],
      threshold: Float,
      trimWhitespace: Boolean): Seq[SentenceSpan] = {

    val spans = scala.collection.mutable.ArrayBuffer.empty[SentenceSpan]
    val n = text.length
    var sentStart = 0

    for (i <- 0 until n) {
      if (charProbs(i) >= threshold) {
        addSpan(text, sentStart, i + 1, trimWhitespace, spans)
        sentStart = i + 1
      }
    }
    if (sentStart < n) addSpan(text, sentStart, n, trimWhitespace, spans)

    spans.toSeq
  }

  /** Append `text[rawStart, rawEnd)` as a span (optionally whitespace-trimmed), skipping it if it
    * collapses to empty. The stored `end` is inclusive, per Spark NLP's annotation convention.
    */
  private def addSpan(
      text: String,
      rawStart: Int,
      rawEnd: Int,
      trim: Boolean,
      buf: scala.collection.mutable.ArrayBuffer[SentenceSpan]): Unit = {

    var s = rawStart
    var e = rawEnd
    if (trim) {
      while (s < e && text.charAt(s).isWhitespace) s += 1
      while (e > s && text.charAt(e - 1).isWhitespace) e -= 1
    }
    if (s < e) buf += SentenceSpan(begin = s, end = e - 1, text = text.substring(s, e))
  }


  /** Reshape a flat row-major array into a `rows x cols` 2-D array for ONNX tensor creation. */
  private def to2DLong(flat: Array[Long], rows: Int, cols: Int): Array[Array[Long]] =
    Array.tabulate(rows)(r => flat.slice(r * cols, (r + 1) * cols))

  private def to2DFloat(flat: Array[Float], rows: Int, cols: Int): Array[Array[Float]] =
    Array.tabulate(rows)(r => flat.slice(r * cols, (r + 1) * cols))



}
