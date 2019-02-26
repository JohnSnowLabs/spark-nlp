package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

/**
  * structure representing a sentence and its boundaries
  */
case class Sentence(content: String, start: Int, end: Int)

object Sentence {
  def fromTexts(texts: String*): Seq[Sentence] = {
    var idx = 0
    texts.map{ text =>
      val sentence = Sentence(text, idx, idx + text.length - 1)
      idx += text.length + 1
      sentence
    }
  }
}

/**
  * Helper object to work work with Sentence
  */
object SentenceSplit extends Annotated[Sentence] {
  override def annotatorType: String = AnnotatorType.DOCUMENT

  override def unpack(annotations: Seq[Annotation]): Seq[Sentence] = {
    annotations.filter(_.annotatorType == annotatorType)
      .map(annotation =>
        Sentence(annotation.result, annotation.begin, annotation.end)
      )
  }

  override def pack(items: Seq[Sentence]): Seq[Annotation] = {
    items
      .sortBy(i => i.start)
      .map(item => Annotation(annotatorType, item.start, item.end, item.content, Map.empty[String, String]))
  }
}

/**
  * Helper object to work work with Chunks
  */
object ChunkSplit extends Annotated[Sentence] {
  override def annotatorType: String = AnnotatorType.CHUNK

  override def unpack(annotations: Seq[Annotation]): Seq[Sentence] = {
    annotations.filter(_.annotatorType == annotatorType)
      .map(annotation =>
        Sentence(annotation.result, annotation.begin, annotation.end)
      )
  }

  override def pack(items: Seq[Sentence]): Seq[Annotation] = {
    items.map(item => Annotation(annotatorType, item.start, item.end, item.content, Map.empty[String, String]))
  }
}
