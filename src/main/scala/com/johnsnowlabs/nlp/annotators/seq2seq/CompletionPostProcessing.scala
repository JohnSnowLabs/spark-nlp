package com.johnsnowlabs.nlp.annotators.seq2seq

import org.apache.spark.ml.param.{Param, Params}

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

  protected def processCompletions(results: Array[String]): Array[String] = {
    getRemoveThinkingTag match {
      case Some(thinkingTag) =>
        val closedTag = s"(?s)<$thinkingTag>.*?</$thinkingTag>"
        // generation can be cut off (nPredict) before the closing tag appears; in that
        // case fall back to stripping from the unclosed opening tag to end-of-string,
        // otherwise the raw in-progress reasoning leaks through unstripped.
        val unclosedTag = s"(?s)<$thinkingTag>.*"
        results.map { text =>
          val closedStripped = text.replaceFirst(closedTag, "")
          val stripped =
            if (closedStripped == text) closedStripped.replaceFirst(unclosedTag, "")
            else closedStripped
          stripped.trim
        }
      case None => results
    }
  }
}
