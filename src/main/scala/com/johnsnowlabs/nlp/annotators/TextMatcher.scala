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

class TextMatcher(override val uid: String) extends AnnotatorApproach[TextMatcherModel] with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN)

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val description: String = "Extracts entities from target dataset given in a text file"

  val entities = new ExternalResourceParam(this, "entities", "entities external resource.")
  val caseSensitive = new BooleanParam(this, "caseSensitive", "whether to match regardless of case. Defaults true")
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")
  val tokenizer = new StructFeature[TokenizerModel](this, "tokenizer")

  setDefault(inputCols,Array(TOKEN))
  setDefault(caseSensitive, true)
  setDefault(mergeOverlapping, false)

  def setEntities(value: ExternalResource): this.type =
    set(entities, value)

  def setEntities(path: String, readAs: ReadAs.Format, options: Map[String, String] = Map("format" -> "text")): this.type =
    set(entities, ExternalResource(path, readAs, options))

  def setTokenizer(tokenizer: TokenizerModel): this.type = set(this.tokenizer, tokenizer)

  def getTokenizer: TokenizerModel = $$(tokenizer)

  def setCaseSensitive(v: Boolean): this.type =
    set(caseSensitive, v)

  def getCaseSensitive: Boolean =
    $(caseSensitive)

  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

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