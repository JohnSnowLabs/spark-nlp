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
        TextSentence(annotation.result, annotation.metadata(Annotation.BEGIN).toInt, annotation.metadata(Annotation.END).toInt)
      )
  }

  override def pack(items: Seq[TextSentence]): Seq[Annotation] = {
    items.map(item => Annotation(
      annotatorType,
      item.text,
      Map(annotatorType -> item.text, Annotation.BEGIN -> item.begin.toString, Annotation.END -> item.end.toString))
    )
  }
}


object Tokenized extends Annotated[TokenizedSentence] {

  override def annotatorType = AnnotatorType.TOKEN

  override def unpack(annotations: Seq[Annotation]): Seq[TokenizedSentence] = {
    val tokens = annotations
      .filter(_.annotatorType == annotatorType)
      .toArray
      .sortBy(a => a.metadata(Annotation.BEGIN).toInt)

    val tokenBegin = tokens.map(t => t.metadata(Annotation.BEGIN).toInt)
    val tokenEnd = tokens.map(t => t.metadata(Annotation.END).toInt)

    def find(begin: Int, end: Int): Array[IndexedToken] = {
      import scala.collection.Searching._
      val beginIdx = tokenBegin.search(begin).insertionPoint
      val endIdx = tokenEnd.search(end + 1).insertionPoint

      val result = Array.fill[IndexedToken](endIdx - beginIdx)(null)
      for (i <- beginIdx until endIdx) {
        val token = tokens(i)
        result(i - beginIdx) = IndexedToken(token.result, token.metadata(Annotation.BEGIN).toInt, token.metadata(Annotation.END).toInt)
      }

      result
    }

    SentenceSplit.unpack(annotations).map(sentence =>
        new TokenizedSentence(find(sentence.begin, sentence.end)))
  }

  override def pack(items: Seq[TokenizedSentence]): Seq[Annotation] = {
    items.flatMap(sentence => sentence.indexedTokens.map(token =>
      new Annotation(
        annotatorType,
        token.token,
        Map(annotatorType -> token.token, Annotation.BEGIN -> token.begin.toString, Annotation.END -> token.end.toString))))
  }
}

trait Tagged[T >: TaggedSentence <: TaggedSentence] extends Annotated[T] {
  val emptyTag = ""

  override def unpack(annotations: Seq[Annotation]): Seq[T] = {

    val tokenized = Tokenized.unpack(annotations)
    val tagAnnotations = annotations
      .filter(a => a.annotatorType == annotatorType)
      .sortBy(a => a.metadata(Annotation.BEGIN).toInt)
      .toIterator

    var annotation: Option[Annotation] = None

    tokenized.map { sentence =>
      val tokens = sentence.indexedTokens.map { token =>
        while (tagAnnotations.hasNext && (annotation.isEmpty || annotation.get.metadata(Annotation.BEGIN).toInt < token.begin))
          annotation = Some(tagAnnotations.next)

        val tag = if (annotation.isDefined && annotation.get.metadata(Annotation.BEGIN).toInt == token.begin)
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
      new Annotation(
        annotatorType,
        tag.tag,
        Map("tag" -> tag.tag, "word" -> tag.word, Annotation.BEGIN -> tag.begin.toString, Annotation.END -> tag.end.toString))
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
    val position2Tag = labelAnnotations.map(a => (a.metadata(Annotation.BEGIN).toInt, a.metadata(Annotation.END).toInt) -> a.metadata("tag")).toMap

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
