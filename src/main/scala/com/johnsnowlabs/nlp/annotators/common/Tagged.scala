package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{NAMED_ENTITY, POS}
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import scala.util.Random


trait Tagged[T >: TaggedSentence <: TaggedSentence] extends Annotated[T] {
  val emptyTag = "O"

  override def unpack(annotations: Seq[Annotation]): Seq[T] = {

    val tokenized = TokenizedWithSentence.unpack(annotations)

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
          annotation.get.result
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
        tag.begin,
        tag.end,
        tag.tag,
        Map("word" -> tag.word) ++ {
          if (tag.confidence.isDefined)
            Map("confidence" -> tag.confidence.get.toString)
          else Map.empty[String, String]
        })
    ))
  }

  /**
    * Method is usefull for testing.
    * It's possible to collect:
    * - correct labels TextSentenceLabels
    * - and model prediction NerTaggedSentence
    */
  def collectLabeledInstances(dataset: Dataset[Row],
                          taggedCols: Seq[String],
                          labelColumn: String): Array[(TextSentenceLabels, T)] = {

    dataset
      .select(labelColumn, taggedCols:_*)
      .collect()
      .flatMap{row =>
        val labelAnnotations = getAnnotations(row, 0)
        val sentenceAnnotations = (1 to taggedCols.length).flatMap(idx => getAnnotations(row, idx))
        val sentences = unpack(sentenceAnnotations)
        val labels = getLabelsFromTaggedSentences(sentences, labelAnnotations)
        labels.zip(sentences)
      }
  }

  def getAnnotations(row: Row, colNum: Int): Seq[Annotation] = {
    row.getAs[Seq[Row]](colNum).map(obj => Annotation(obj))
  }


  protected def getLabelsFromSentences(sentences: Seq[WordpieceEmbeddingsSentence],
                                       labelAnnotations: Seq[Annotation]): Seq[TextSentenceLabels] = {
    val sortedLabels = labelAnnotations.sortBy(a => a.begin).toArray

    sentences.map{sentence =>
      // Extract labels only for wordpiece that are at the begin of tokens
      val tokens = sentence.tokens.filter(t => t.isWordStart)
      val labels = tokens.map { w =>
        val tag = Annotation.searchCoverage(sortedLabels, w.begin, w.end)
          .map(a => a.result)
          .headOption
          .getOrElse(emptyTag)

        tag
      }
      TextSentenceLabels(labels)
    }
  }


  protected def getLabelsFromTaggedSentences(sentences: Seq[TaggedSentence], labelAnnotations: Seq[Annotation]): Seq[TextSentenceLabels] = {
    val sortedLabels = labelAnnotations.sortBy(a => a.begin).toArray

    sentences.map{sentence =>
      val labels = sentence.indexedTaggedWords.map { w =>
        val tag = Annotation.searchCoverage(sortedLabels, w.begin, w.end)
          .map(a => a.result)
          .headOption
          .getOrElse(emptyTag)

        tag
      }
      TextSentenceLabels(labels)
    }
  }
}

object PosTagged extends Tagged[PosTaggedSentence]{
  override def annotatorType: String = POS
}

object NerTagged extends Tagged[NerTaggedSentence]{
  override def annotatorType: String = NAMED_ENTITY

  def collectTrainingInstancesWithPos(dataset: Dataset[Row],
                                      posTaggedCols: Seq[String],
                                      labelColumn: String):
  Array[(TextSentenceLabels, PosTaggedSentence, WordpieceEmbeddingsSentence)] = {

    val annotations = dataset
      .select(labelColumn, posTaggedCols:_*)
      .collect()

    annotations
      .flatMap{row =>
        val labelAnnotations = this.getAnnotations(row, 0)
        val sentenceAnnotations  = (1 to posTaggedCols.length).flatMap(idx => getAnnotations(row, idx))
        val sentences = PosTagged.unpack(sentenceAnnotations)
          .filter(s => s.indexedTaggedWords.nonEmpty)
          .sortBy(s => s.indexedTaggedWords.head.begin)

        val withEmbeddings = WordpieceEmbeddingsSentence.unpack(sentenceAnnotations)
          .filter(s => s.tokens.nonEmpty)
          .sortBy(s => s.tokens.head.begin)

        val labels = getLabelsFromTaggedSentences(sentences, labelAnnotations)
        labels.zip(sentences zip withEmbeddings)
          .map{case (l, (s, w)) => (l, s, w)}
      }
  }



  /** FIXME: ColNums not always in the given order*/
  def iterateOnDataframe(dataset: Dataset[Row],
                         sentenceCols: Seq[String],
                         labelColumn: String,
                         batchSize:Int): Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {


      new Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] {

        import com.johnsnowlabs.nlp.annotators.common.DatasetHelpers._

        // Send batches, don't collect(), only keeping a single batch in memory anytime
        val it = dataset
          .select(labelColumn, sentenceCols: _*)
          .randomize // to improve training
          .toLocalIterator()

        // create a batch
        override def next(): Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)] = {
          var count = 0
          var thisBatch = Array.empty[(TextSentenceLabels, WordpieceEmbeddingsSentence)]

          while (it.hasNext && count < batchSize) {
            count += 1
            val nextRow = it.next

            val labelAnnotations = getAnnotations(nextRow, 0)
            val sentenceAnnotations = (1 to sentenceCols.length).flatMap(idx => getAnnotations(nextRow, idx))
            val sentences = WordpieceEmbeddingsSentence.unpack(sentenceAnnotations)
            val labels = getLabelsFromSentences(sentences, labelAnnotations)
            val thisOne = labels.zip(sentences)

            thisBatch = thisBatch ++ thisOne
          }
          thisBatch
        }

        override def hasNext: Boolean = it.hasNext
      }

  }



  /** FIXME: ColNums not always in the given order*/
  def interateOnArray(inputArray: Array[Row],
                      sentenceCols: Seq[String],
                      labelColumn: String,
                      batchSize:Int): Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {
    import com.johnsnowlabs.nlp.annotators.common.DatasetHelpers._

     slice(Random.shuffle(inputArray.toSeq)
      .flatMap { row =>
        val labelAnnotations = this.getAnnotations(row, 0)
        val sentenceAnnotations = (1 to sentenceCols.length).flatMap(idx => getAnnotations(row, idx))
        val sentences = WordpieceEmbeddingsSentence.unpack(sentenceAnnotations)
        val labels = getLabelsFromSentences(sentences, labelAnnotations)
        labels.zip(sentences)
      }, batchSize)
    }
}
