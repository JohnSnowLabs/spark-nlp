package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.{IndexedTaggedWord, IndexedToken, TaggedSentence, TokenizedSentence}
import org.apache.spark.sql.{Dataset, Row}
import Annotated._

trait Annotated[TResult] {
  def annotatorType: String

  def unpack(annotations: Seq[Annotation]): Seq[TResult]

  def pack(items: Seq[TResult]): Seq[Annotation]
}

object Annotated {
  type PosTaggedSentence = TaggedSentence
  type NerTaggedSentence = TaggedSentence
}

case class TextSentence(text: String, begin: Int, end: Int)

object SentenceSplit extends Annotated[TextSentence] {
  override def annotatorType: String = AnnotatorType.DOCUMENT

  override def unpack(annotations: Seq[Annotation]): Seq[TextSentence] = {
    annotations.filter(_.annotatorType == annotatorType)
      .map(annotation =>
        TextSentence(annotation.metadata(annotatorType), annotation.begin, annotation.end)
      )
  }

  override def pack(items: Seq[TextSentence]): Seq[Annotation] = {
    items.map(item => Annotation(annotatorType, item.begin, item.end, Map(annotatorType -> item.text)))
  }
}


object Tokenized extends Annotated[TokenizedSentence] {

  override def annotatorType = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray
      .sortBy(a => a.begin)

    val tokenBegin = tokens.map(t => t.begin)
    val tokenEnd = tokens.map(t => t.end)

    def find(begin: Int, end: Int): Array[IndexedToken] = {
      import scala.collection.Searching._
      val beginIdx = tokenBegin.search(begin).insertionPoint
      val endIdx = tokenEnd.search(end + 1).insertionPoint

      val result = Array.fill[IndexedToken](endIdx - beginIdx)(null)
      for (i <- beginIdx until endIdx) {
        val token = tokens(i)
        result(i - beginIdx) = IndexedToken(token.metadata(annotatorType), token.begin, token.end)
      }

      result
    }

    SentenceSplit.unpack(annotations).map(sentence =>
        new TokenizedSentence(find(sentence.begin, sentence.end)))
  }

  override def pack(items: Seq[TokenizedSentence]): Seq[Annotation] = {
    items.flatMap(sentence => sentence.indexedTokens.map(token =>
      new Annotation(annotatorType, token.begin, token.end, Map(annotatorType -> token.token))))
  }
}

trait Tagged[T >: TaggedSentence <: TaggedSentence] extends Annotated[T] {
  val emptyTag = ""

  override def unpack(annotations: Seq[Annotation]): Seq[T] = {

    val tokenized = Tokenized.unpack(annotations)
    val tagAnnotations = annotations
      .filter(a => a.annotatorType == annotatorType)
      .sortBy(a => a.begin)
      .toIterator

    var annotation: Option[Annotation] = None

    tokenized.map { sentence =>
      val tokens = sentence.indexedTokens.map { token =>
        while (tagAnnotations.hasNext && (annotation.isEmpty || annotation.get.begin < token.begin))
          annotation = Some(tagAnnotations.next)

        val tag = if (annotation.isDefined && annotation.get.begin == token.begin)
          annotation.get.metadata("tag")
        else
          emptyTag

        IndexedTaggedWord(token.token, tag, token.begin, token.end)
      }

      new TaggedSentence(tokens)
    }
  }

  override def pack(items: Seq[T]): Seq[Annotation] = {
    items.flatMap(item => item.indexedTaggedWords.map(tag =>
      new Annotation(annotatorType, tag.begin, tag.end, Map("tag" -> tag.tag, "word" -> tag.word))
    ))
  }
}

object PosTagged extends Tagged[PosTaggedSentence]{
  override def annotatorType: String = POS
}

object NerTagged extends Tagged[NerTaggedSentence]{
  override def annotatorType: String = NAMED_ENTITY

  def collectNerInstances(dataset: Dataset[Row],
                          nerTaggedCols: Seq[String],
                          labelColumn: String): Array[(TextSentenceLabels, NerTaggedSentence)] = {

    dataset
      .select(labelColumn, nerTaggedCols:_*)
      .collect()
      .flatMap{row =>
        val labelAnnotations = getAnnotations(row, 0)
        val sentenceAnnotations = (1 to nerTaggedCols.length).flatMap(idx => getAnnotations(row, idx))
        val sentences = unpack(sentenceAnnotations)
        val labels = getLabels(sentences, labelAnnotations)
        labels.zip(sentences)
      }
  }

  def collectTrainingInstances(dataset: Dataset[Row],
                               posTaggedCols: Seq[String],
                               labelColumn: String): Array[(TextSentenceLabels, PosTaggedSentence)] = {

    dataset
      .select(labelColumn, posTaggedCols:_*)
      .collect()
      .flatMap{row =>
        val labelAnnotations = getAnnotations(row, 0)
        val sentenceAnnotations = (1 to posTaggedCols.length).flatMap(idx => getAnnotations(row, idx))
        val sentences = PosTagged.unpack(sentenceAnnotations)
        val labels = getLabels(sentences, labelAnnotations)
        labels.zip(sentences)
      }
  }

  private def getLabels(sentences: Seq[TaggedSentence], labelAnnotations: Seq[Annotation]): Seq[TextSentenceLabels] = {
    val position2Tag = labelAnnotations.map(a => (a.begin, a.end) -> a.metadata("tag")).toMap

    sentences.map{sentence =>
      val labels = sentence.indexedTaggedWords.map { w =>
        val tag = position2Tag.get((w.begin, w.end))
        tag.getOrElse("")
      }
      TextSentenceLabels(labels)
    }
  }

  private def getAnnotations(row: Row, colNum: Int): Seq[Annotation] = {
    row.getAs[Seq[Row]](colNum).map(obj => Annotation(obj))
  }
}
