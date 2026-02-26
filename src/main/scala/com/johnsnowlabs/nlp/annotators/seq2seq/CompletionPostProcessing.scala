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
        results.map(text => text.replaceFirst(s"(?s)<$thinkingTag>.*?</$thinkingTag>", "").trim)
      case None => results
    }
  }
}
