package com.johnsnowlabs.nlp.annotators.seq2seq

import org.apache.spark.ml.param.{Param, Params}

/** One post-processed completion: the cleaned text, plus where that text starts in the raw
  * completion the model actually generated.
  *
  * @param text
  *   the cleaned completion, as it will appear in the output annotation's `result`
  * @param beginOffset
  *   the character offset at which `text` begins inside the raw completion, or `None` when the
  *   cleaned text is not a contiguous slice of it (so per-token character offsets from
  *   `completion_probabilities` cannot be mapped onto it at all)
  */
private[nlp] case class ProcessedCompletion(text: String, beginOffset: Option[Int])

private[nlp] trait CompletionPostProcessing {
  this: Params =>

  /** @group param */
  val removeThinkingTag =
    new Param[String](
      this,
      "removeThinkingTag",
      "Set a thinking tag (e.g. think) to be removed from output. Will match <TAG>...</TAG>")

  /** Set a thinking tag (e.g. `think`) to be removed from output. Will produce the regex
    * `(?s)<$TAG>.+?</$TAG>`
    * @group setParam
    */
  def setRemoveThinkingTag(value: String): this.type = set(removeThinkingTag, value)

  /** @group getParam */
  def getRemoveThinkingTag: Option[String] = get(removeThinkingTag)

  /** Narrows `raw` to `[from, until)` and trims whitespace off both ends, reporting where the
    * result starts in `raw`.
    */
  private def sliceTrimmed(raw: String, from: Int, until: Int): ProcessedCompletion = {
    var begin = from
    var end = until
    while (begin < end && raw.charAt(begin).isWhitespace) begin += 1
    while (end > begin && raw.charAt(end - 1).isWhitespace) end -= 1
    ProcessedCompletion(raw.substring(begin, end), Some(begin))
  }

  private def processOne(raw: String): ProcessedCompletion = getRemoveThinkingTag match {
    case None => ProcessedCompletion(raw, Some(0))
    case Some(thinkingTag) =>
      val closedTag = s"(?s)<$thinkingTag>.*?</$thinkingTag>".r
      // generation can be cut off (nPredict) before the closing tag appears; in that case fall
      // back to stripping from the unclosed opening tag to end-of-string, otherwise the raw
      // in-progress reasoning leaks through unstripped.
      val unclosedTag = s"(?s)<$thinkingTag>.*".r
      closedTag.findFirstMatchIn(raw).orElse(unclosedTag.findFirstMatchIn(raw)) match {
        // The overwhelmingly common shape: the block leads the completion (possibly after some
        // whitespace, which trimming would discard anyway), so what survives is a contiguous
        // suffix of it.
        case Some(m) if raw.substring(0, m.start).forall(_.isWhitespace) =>
          sliceTrimmed(raw, m.end, raw.length)
        case Some(m) if raw.substring(m.end).forall(_.isWhitespace) =>
          sliceTrimmed(raw, 0, m.start)
        case Some(m) =>
          // A block with real text on both sides leaves two disjoint pieces, so no single offset
          // maps completion_probabilities spans onto the result.
          ProcessedCompletion((raw.substring(0, m.start) + raw.substring(m.end)).trim, None)
        case None => sliceTrimmed(raw, 0, raw.length)
      }
  }

  /** Cleans each completion, reporting where the cleaned text starts in the raw completion so
    * callers can keep per-token metadata aligned with it.
    */
  protected def processCompletionsWithOffsets(
      results: Array[String]): Array[ProcessedCompletion] =
    results.map(processOne)

  protected def processCompletions(results: Array[String]): Array[String] =
    processCompletionsWithOffsets(results).map(_.text)
}
