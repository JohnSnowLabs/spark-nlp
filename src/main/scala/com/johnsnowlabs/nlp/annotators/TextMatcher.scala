package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.collections.SearchTrie
import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, _}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, ParamsAndFeaturesWritable}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset


/** Annotator to match entire phrases (by token) provided in a file against a Document
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/TextMatcherTestSpec.scala]] for reference on how to use this API
  *
  * @param uid internal uid required to generate writable annotators
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  * */
class TextMatcher(override val uid: String) extends AnnotatorApproach[TextMatcherModel] with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  /** Output annotator type : DOCUMENT, TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN)
  /** Output annotator type : CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Extracts entities from target dataset given in a text file */
  override val description: String = "Extracts entities from target dataset given in a text file"
  /** entities external resource.
    *
    * @group param
    **/
  val entities = new ExternalResourceParam(this, "entities", "entities external resource.")
  /** whether to match regardless of case. Defaults true
    *
    * @group param
    **/
  val caseSensitive = new BooleanParam(this, "caseSensitive", "whether to match regardless of case. Defaults true")
  /** whether to merge overlapping matched chunks. Defaults false
    *
    * @group param
    **/
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")
  /** Value for the entity metadata field
    *
    * @group param
    **/
  val entityValue = new Param[String](this, "entityValue", "Value for the entity metadata field")
  /** Whether the TextMatcher should take the CHUNK from TOKEN or not
    *
    * @group param
    **/
  val buildFromTokens = new BooleanParam(this, "buildFromTokens", "Whether the TextMatcher should take the CHUNK from TOKEN or not")
  /** Tokenizer
    *
    * @group param
    **/
  val tokenizer = new StructFeature[TokenizerModel](this, "tokenizer")

  setDefault(inputCols, Array(TOKEN))
  setDefault(caseSensitive, true)
  setDefault(mergeOverlapping, false)
  setDefault(entityValue, "entity")
  setDefault(buildFromTokens, false)


  /** Provides a file with phrases to match. Default: Looks up path in configuration.
    *
    * @group getParam
    **/
  def setEntities(value: ExternalResource): this.type =
    set(entities, value)

  /** Provides a file with phrases to match. Default: Looks up path in configuration. */

  /**
    *
    * @param path    a path to a file that contains the entities in the specified format.
    * @param readAs  the format of the file, can be one of {ReadAs.LINE_BY_LINE, ReadAs.SPARK_DATASET}. Defaults to LINE_BY_LINE.
    * @param options a map of additional parameters. Defaults to {“format”: “text”}.
    * @return this
    * @group getParam
    */
  def setEntities(path: String, readAs: ReadAs.Format, options: Map[String, String] = Map("format" -> "text")): this.type =
    set(entities, ExternalResource(path, readAs, options))

  /** @group setParam */
  def setTokenizer(tokenizer: TokenizerModel): this.type = set(this.tokenizer, tokenizer)

  /** @group GetParam */
  def getTokenizer: TokenizerModel = $$(tokenizer)

  /** whether to match regardless of case. Defaults true
    *
    * @group setParam
    **/
  def setCaseSensitive(v: Boolean): this.type = set(caseSensitive, v)

  /** whether to match regardless of case. Defaults true
    *
    * @group getParam
    **/
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** whether to merge overlapping matched chunks. Defaults false
    *
    * @group setParam
    **/
  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  /** whether to merge overlapping matched chunks. Defaults false
    *
    * @group getParam
    **/
  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /** Setter for Value for the entity metadata field
    *
    * @group setParam
    **/
  def setEntityValue(v: String): this.type = set(entityValue, v)

  /** Getter for Value for the entity metadata field
    *
    * @group getParam
    **/
  def getEntityValue: String = $(entityValue)

  /** Setter for buildFromTokens param
    *
    * @group setParam
    **/
  def setBuildFromTokens(v: Boolean): this.type = set(buildFromTokens, v)

  /** Getter for buildFromTokens param
    *
    * @group getParam
    **/
  def getBuildFromTokens: Boolean = $(buildFromTokens)

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
      .setBuildFromTokens($(buildFromTokens))
  }

}

object TextMatcher extends DefaultParamsReadable[TextMatcher]