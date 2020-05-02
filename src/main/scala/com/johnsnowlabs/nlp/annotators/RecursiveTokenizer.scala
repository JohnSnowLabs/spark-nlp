package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.param.StringArrayParam


class RecursiveTokenizer(override val uid: String)
  extends AnnotatorApproach[RecursiveTokenizerModel] with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("SILLY_TOKENIZER"))

  val prefixes = new StringArrayParam(this, "prefixes", "Strings that will be split when found at the beginning of token.")
  def setPrefixes(p: Array[String]):this.type = set(prefixes, p.sortBy(_.size).reverse)

  val suffixes = new StringArrayParam(this, "suffixes", "Strings that will be split when found at the end of token.")
  def setSuffixes(s: Array[String]):this.type = set(suffixes, s.sortBy(_.size).reverse)

  val infixes = new StringArrayParam(this, "infixes", "Strings that will be split when found at the middle of token.")
  def setInfixes(p: Array[String]):this.type = set(infixes, p.sortBy(_.size).reverse)

  val whitelist = new StringArrayParam(this, "whitelist", "Whitelist.")
  def setWhitelist(w: Array[String]):this.type = set(whitelist, w)

  setDefault(infixes,  Array("\n", "(", ")"))
  setDefault(prefixes, Array("'", "\"", "(", "[", "\n"))
  setDefault(suffixes, Array(".", ":", "%", ",", ";", "?", "'", "\"", ")", "]", "\n", "!", "'s"))
  setDefault(whitelist, Array("it's", "that's", "there's", "he's", "she's", "what's", "let's", "who's",
    "It's", "That's", "There's", "He's", "She's", "What's", "Let's", "Who's"))


  override val outputAnnotatorType: AnnotatorType = AnnotatorType.TOKEN

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val description: String = "Simplest possible tokenizer"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): RecursiveTokenizerModel = {
    new RecursiveTokenizerModel().
      setPrefixes(getOrDefault(prefixes)).
      setSuffixes(getOrDefault(suffixes)).
      setInfixes(getOrDefault(infixes)).
      setWhitelist(getOrDefault(whitelist).toSet)
  }
}
