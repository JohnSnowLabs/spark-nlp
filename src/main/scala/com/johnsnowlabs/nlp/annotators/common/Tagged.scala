package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{NAMED_ENTITY, POS}
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import org.apache.spark.sql.{Dataset, Row}


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
