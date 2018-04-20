package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ListBuffer

/** Created by danilo 16/04/2018,
  * Symmetric Delete spelling correction algorithm
  * inspired on https://github.com/wolfgarbe/SymSpell
  * */
class SymmetricDeleteApproach(override val uid: String)
  extends AnnotatorApproach[SymmetricDeleteModel]
    with SymmetricDeleteParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Spell checking algorithm inspired on Symmetric Delete algorithm"

  val corpus = new ExternalResourceParam(this, "corpus", "folder or file with text that teaches about the language")

  setDefault(maxEditDistance, 3)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "spell checker corpus needs 'tokenPattern' regex for tagging words. e.g. [a-zA-Z]+")
    set(corpus, value)
  }

  def setCorpus(path: String,
                tokenPattern: String,
                readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))


  // AnnotatorType shows the structure of the result, we can have annotators with the same result
  override val annotatorType: AnnotatorType = SPELL

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN) //The approach required to work

  def this() = this(Identifiable.randomUID("SPELL")) // constructor required for the annotator to work in python

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SymmetricDeleteModel = {

    val corpusDeriveWordCount: Map[String, (ListBuffer[String], Long)] =
        ResourceHelper.deriveWordCount(er = $(corpus),
                                       p = recursivePipeline,
                                       med = $(maxEditDistance)).toMap
    val longestWordLength = ResourceHelper.getLongestWordLength
    //println("Dictionary created...")

    new SymmetricDeleteModel()
      .setDeriveWordCount(corpusDeriveWordCount)
      .setLongestWordLength(longestWordLength)
  }

}
// This objects reads the class' properties, it enables reding the model after it is stored
object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
