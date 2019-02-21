package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}

/**
  * structure representing a sentence and its boundaries
  */
case class Sentence(content: String, start: Int, end: Int, id: Int)

object Sentence {
  def fromTexts(texts: String*): Seq[Sentence] = {
    var idx = 0
    texts.zipWithIndex.map{ case (text, id) =>
      val sentence = Sentence(text, idx, idx + text.length - 1, id)
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
    annotations.filter(_.annotatorType == annotatorType).zipWithIndex
      .map { case (annotation, id) =>
        Sentence(annotation.result, annotation.begin, annotation.end, id)
      }
  }

  override def pack(items: Seq[Sentence]): Seq[Annotation] = {
    items.map(item => Annotation(annotatorType, item.start, item.end, item.content, Map.empty[String, String]))
  }
}

/**
  * Helper object to work work with Chunks
  */
object ChunkSplit extends Annotated[Sentence] {
  override def annotatorType: String = AnnotatorType.CHUNK

  override def unpack(annotations: Seq[Annotation]): Seq[Sentence] = {
    annotations.filter(_.annotatorType == annotatorType).zipWithIndex
      .map { case (annotation, id) =>
        Sentence(annotation.result, annotation.begin, annotation.end, id)
      }
  }

  override def pack(items: Seq[Sentence]): Seq[Annotation] = {
    items.map(item => Annotation(annotatorType, item.start, item.end, item.content, Map.empty[String, String]))
  }
}
