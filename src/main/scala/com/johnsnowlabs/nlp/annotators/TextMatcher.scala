package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, _}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, ParamsAndFeaturesWritable}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset


/** Annotator to match entire phrases (by token) provided in a file against a Document
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TextMatcherTestSpec.scala]] for reference on how to use this API
  *
  * @param uid internal uid required to generate writable annotators
  **/
class TextMatcher(override val uid: String) extends AnnotatorApproach[TextMatcherModel] with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  /** Output annotator type : DOCUMENT, TOKEN */
  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN)
  /** Output annotator type : CHUNK */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Extracts entities from target dataset given in a text fil */
  override val description: String = "Extracts entities from target dataset given in a text file"
  /** entities external resource. */
  val entities = new ExternalResourceParam(this, "entities", "entities external resource.")
  /** whether to match regardless of case. Defaults true */
  val caseSensitive = new BooleanParam(this, "caseSensitive", "whether to match regardless of case. Defaults true")
  /** whether to merge overlapping matched chunks. Defaults false */
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")
  /** Tokenizer */
  val tokenizer = new StructFeature[TokenizerModel](this, "tokenizer")

  setDefault(inputCols, Array(TOKEN))
  setDefault(caseSensitive, true)
  setDefault(mergeOverlapping, false)


  /** Provides a file with phrases to match. Default: Looks up path in configuration. */
  def setEntities(value: ExternalResource): this.type =
    set(entities, value)

  /** Provides a file with phrases to match. Default: Looks up path in configuration. */

  /**
    *
    * @param path    a path to a file that contains the entities in the specified format.
    * @param readAs  the format of the file, can be one of {ReadAs.LINE_BY_LINE, ReadAs.SPARK_DATASET}. Defaults to LINE_BY_LINE.
    * @param options a map of additional parameters. Defaults to {“format”: “text”}.
    * @return this
    */
  def setEntities(path: String, readAs: ReadAs.Format, options: Map[String, String] = Map("format" -> "text")): this.type =
    set(entities, ExternalResource(path, readAs, options))

  def setTokenizer(tokenizer: TokenizerModel): this.type = set(this.tokenizer, tokenizer)

  def getTokenizer: TokenizerModel = $$(tokenizer)

  /** whether to match regardless of case. Defaults true */
  def setCaseSensitive(v: Boolean): this.type = set(caseSensitive, v)

  /** whether to match regardless of case. Defaults true */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** whether to merge overlapping matched chunks. Defaults false */
  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  /** whether to merge overlapping matched chunks. Defaults false */
  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /**
    * Loads entities from a provided source.
    */
  private def loadEntities(dataset: Dataset[_]): Array[Array[String]] = {
    val phrases: Array[String] = ResourceHelper.parseLines($(entities))
    val parsedEntities: Array[Array[String]] = {
      get(tokenizer) match {
        case Some(tokenizerModel: TokenizerModel) =>
          phrases.map {
            line =>
              val annotation = Seq(Annotation(line))
              val tokens = tokenizerModel.annotate(annotation)
              tokens.map(_.result).toArray
          }
        case _ =>
          phrases.map {
            line =>
              line.split(" ")
          }
      }
    }
    parsedEntities
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): TextMatcherModel = {
    new TextMatcherModel()
      .setSearchTrie(SearchTrie.apply(loadEntities(dataset), $(caseSensitive)))
      .setMergeOverlapping($(mergeOverlapping))
  }

}

object TextMatcher extends DefaultParamsReadable[TextMatcher]