package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence

import scala.collection.mutable.ArrayBuffer


/**
  * Works with different NER representations as tags
  * Supports: IOB and IOB2 https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
  */
object NerTagsEncoding {

  /**
    * Converts from IOB or IOB2 to list of NamedEntity
    * @param tagged Sentences of IOB of IOB2 tagged sentences
    * @param doc Source doc text
    * @return Extracted Named Entities
    */
  def fromIOB(tagged: Seq[NerTaggedSentence], doc: String): Seq[NamedEntity] = {
    val result = ArrayBuffer[NamedEntity]()

    for (sentence <- tagged) {
      val words = sentence.words.length

      var lastTag: Option[String] = None
      var lastTagStart = -1

      def flushEntity(startIdx: Int, endIdx: Int): Unit = {
        val start = sentence.indexedTaggedWords(startIdx).begin
        val end = sentence.indexedTaggedWords(endIdx).end
        val entity = NamedEntity(start, end, lastTag.get, doc.substring(start, end + 1))

        result.append(entity)
        lastTag = None
      }

      for (i <- 0 until words) {
        val tag = sentence.tags(i)
        if (lastTag.isDefined && (tag.startsWith("B-") || tag == "O")) {
          flushEntity(lastTagStart, i - 1)
        }

        if (lastTag.isEmpty && tag != "O") {
          lastTag = Some(tag.substring(2))
          lastTagStart = i
        }
      }

      if (lastTag.isDefined) {
        flushEntity(lastTagStart, words - 1)
      }
    }

    result.toList
  }

}

case class NamedEntity(start: Int, end: Int, entity: String, text: String)
