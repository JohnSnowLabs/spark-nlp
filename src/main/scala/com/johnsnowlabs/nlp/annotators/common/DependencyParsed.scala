package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}


case class DependencyParsedSentence(tokens: Array[WordWithDependency])

case class WordWithDependency(word: String, begin: Int, end: Int, head: Int)


object DependencyParsed extends Annotated[DependencyParsedSentence]{
  override def annotatorType: String = AnnotatorType.DEPENDENCY

  override def unpack(annotations: Seq[Annotation]): Seq[DependencyParsedSentence] = {
    val sentences = TokenizedWithSentence.unpack(annotations)
    val depAnnotations = annotations
      .filter(a => a.annotatorType == annotatorType)
      .sortBy(a => a.begin)

    var last = 0
    sentences.map{sentence =>
      val sorted = sentence.indexedTokens.sortBy(t => t.begin)
      val dependencies = (last until (last + sorted.length)).map { i =>
        depAnnotations(i).metadata("head").toInt
      }

      last += sorted.length

      val words = sorted.zip(dependencies).map{
        case (token, dependency) =>
          WordWithDependency(token.token, token.begin, token.end, dependency)
      }

      DependencyParsedSentence(words)
    }
  }

  override def pack(items: Seq[DependencyParsedSentence]): Seq[Annotation] = {
    items.flatMap{sentence =>
      val sizeSentence = sentence.tokens.length
      sentence.tokens.map { token =>
        val headWord = getHeadWord(token.head, sentence)
        val word = token.word
        val relatedWords = s"($headWord,$word)"
        val realHead = updateHeadsWithRootIndex(token.head, sizeSentence)
        Annotation(annotatorType, token.begin, token.end, relatedWords, Map("head" -> realHead.toString))
      }
    }
  }

  def getHeadWord(head: Int, sentence: DependencyParsedSentence): String = {
    var headWord = "ROOT"
    val sizeSentence = sentence.tokens.length
    if (head != sizeSentence) {
      headWord = sentence.tokens(head).word
    }
    headWord
  }

  def updateHeadsWithRootIndex(head: Int, sizeSentence: Int): Int = {
    var newHead = 0
    if (head != sizeSentence) {
      newHead = head + 1
    }
    newHead
  }

}