package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, TaggedSentence}

import scala.collection.mutable.ArrayBuffer

object CoNLLHelper {

  case class CoNLLTokenCols(uPosTokens: IndexedTaggedWord, xPosTokens: IndexedTaggedWord, lemma: IndexedTaggedWord,
                            sentenceIndex: Int)
  case class CoNLLSentenceCols(uPos: TaggedSentence, xPos: TaggedSentence, lemma: TaggedSentence)

  def readLines(lines: Array[String], explodeSentences: Boolean): Seq[CoNLLUDocument] = {

    val doc = new StringBuilder()
    val lastSentence = ArrayBuffer.empty[CoNLLTokenCols]
    val sentences = ArrayBuffer.empty[CoNLLSentenceCols]

    def addSentence(): Unit = {
      val uPosTokens = clearTokens(lastSentence.map(t => t.uPosTokens).toArray)
      val xPosTokens = clearTokens(lastSentence.map(t => t.xPosTokens).toArray)
      val lemmaTokens = clearTokens(lastSentence.map(t => t.lemma).toArray)
      val uPos = TaggedSentence(uPosTokens)
      val xPos = TaggedSentence(xPosTokens)
      val lemma = TaggedSentence(lemmaTokens)
      val taggedCoNLLSentence = CoNLLSentenceCols(uPos, xPos, lemma)

      sentences.append(taggedCoNLLSentence)
      lastSentence.clear()
    }

    def closeDocument: Option[(String, List[CoNLLSentenceCols])] = {

      val result = (doc.toString, sentences.toList)
      doc.clear()
      sentences.clear()

      if (result._1.nonEmpty) Some(result._1, result._2) else None
    }

    def processCoNLLRow(items: Array[String]): Option[(String, List[CoNLLSentenceCols])] = {
      if (doc.nonEmpty && !doc.endsWith(System.lineSeparator()) && items(3) != "PUNCT")
        doc.append(" ")
      val indexedTaggedCoNLL = getIndexedTaggedCoNLL(items, doc)
      lastSentence.append(indexedTaggedCoNLL)
      None
    }

    def processNewLine(): Option[(String, List[CoNLLSentenceCols])] = {
      if (!explodeSentences && (doc.nonEmpty && !doc.endsWith(System.lineSeparator) && lastSentence.nonEmpty)) {
        doc.append(System.lineSeparator * 2)
      }
      addSentence()
      if (explodeSentences) closeDocument else None
    }

    def processComment(items: Array[String]): Option[(String, List[CoNLLSentenceCols])] = {
      if (items(CoNLLUCols.ID.id).contains("newdoc")) {
        closeDocument
      } else None
    }

    val docs = lines
      .flatMap{ line =>
        val items = line.trim.split("\\t")
        val id = if (items(CoNLLUCols.ID.id).isEmpty) "" else items(CoNLLUCols.ID.id).head.toString

        val coNLLRow = id match {
          case "#" => processComment(items)
          case "" => processNewLine()
          case _ => processCoNLLRow(items)
        }
        coNLLRow
     }

    addSentence()

    val last = if (doc.nonEmpty) Seq((doc.toString, sentences.toList)) else Seq.empty

   (docs ++ last).map{ case(text, textSentence) =>
      val uPos = textSentence.map(t => t.uPos)
      val xPos = textSentence.map(t => t.xPos)
      val lemma = textSentence.map(t => t.lemma)
      CoNLLUDocument(text, uPos, xPos, lemma)
    }
  }

  private def clearTokens(tokens: Array[IndexedTaggedWord]): Array[IndexedTaggedWord] = {
    tokens.filter(t => t.word.trim().nonEmpty)
  }

  private def getIndexedTaggedCoNLL(items: Array[String], doc: StringBuilder, sentenceIndex: Int = 0): CoNLLTokenCols = {
    val begin = doc.length
    doc.append(items(CoNLLUCols.FORM.id))
    val end = doc.length - 1
    val word = items(CoNLLUCols.FORM.id)
    val uPosTag = items(CoNLLUCols.UPOS.id)
    val xPosTag = items(CoNLLUCols.XPOS.id)
    val lemmaValue = items(CoNLLUCols.LEMMA.id)

    val uPos = IndexedTaggedWord(word, uPosTag, begin, end)
    val xPos = IndexedTaggedWord(word, xPosTag, begin, end)
    val lemma = IndexedTaggedWord(lemmaValue, "", begin, end)

    CoNLLTokenCols(uPos, xPos, lemma, sentenceIndex)
  }

}
