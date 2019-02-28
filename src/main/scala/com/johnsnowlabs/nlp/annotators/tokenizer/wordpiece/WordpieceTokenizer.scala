package com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}


class WordpieceTokenizer(override val uid: String) extends AnnotatorApproach[WordpieceTokenizerModel] {

  val vocabulary = new ExternalResourceParam(this, "vocabulary", "Vocabulary that is used by model")
  val lowercase = new BooleanParam(this, name="lowercase", "Convert to lowercase or not")

  setDefault(lowercase, true)

  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  def setVocabulary(path: String,
                           readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                           options: Map[String, String] = Map("format" -> "text")): this.type =
    set(vocabulary, ExternalResource(path, readAs, options))


  def this() = this(Identifiable.randomUID("WORDPIECE_TOKENIZER"))

  override val description: String = "Wordpiece tokenizer"
  override val outputAnnotatorType: AnnotatorType = WORDPIECE

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordpieceTokenizerModel = {
    val words = ResourceHelper.parseLines($(vocabulary)).zipWithIndex.toMap
    new WordpieceTokenizerModel()
      .setVocabulary(words)
      .setLowercase($(lowercase))
  }
}

object WordpieceTokenizer extends DefaultParamsReadable[WordpieceTokenizer]
