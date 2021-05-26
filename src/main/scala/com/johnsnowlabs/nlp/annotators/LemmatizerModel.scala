package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import org.apache.spark.ml.util.Identifiable


/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  *
  * Retrieves lemmas out of words with the objective of returning a base dictionary word. Retrieves the significant part of a word
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/LemmatizerTestSpec.scala]] for examples of how to use this API
  *
  * @param uid required internal uid provided by constructor
  * @@ lemmaDict: A dictionary of predefined lemmas must be provided
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
  */
class LemmatizerModel(override val uid: String) extends AnnotatorModel[LemmatizerModel] with HasSimpleAnnotate[LemmatizerModel] {

  /** Output annotator type : TOKEN
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = TOKEN
  /** Input annotator type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** lemmaDict
    *
    * @group param
    **/
  val lemmaDict: MapFeature[String, String] = new MapFeature(this, "lemmaDict")

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  def setLemmaDict(value: Map[String, String]): this.type = set(lemmaDict, value)

  /**
    * @return one to one annotation from token to a lemmatized word, if found on dictionary or leave the word as is
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { tokenAnnotation =>
      val token = tokenAnnotation.result
      Annotation(
        outputAnnotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        $$(lemmaDict).getOrElse(token, token),
        tokenAnnotation.metadata
      )
    }
  }

}

trait ReadablePretrainedLemmatizer extends ParamsAndFeaturesReadable[LemmatizerModel] with HasPretrained[LemmatizerModel] {
  override val defaultModelName = Some("lemma_antbnc")

  /** Java compliant-overrides */
  override def pretrained(): LemmatizerModel = super.pretrained()
  override def pretrained(name: String): LemmatizerModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): LemmatizerModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): LemmatizerModel = super.pretrained(name, lang, remoteLoc)
}

object LemmatizerModel extends ReadablePretrainedLemmatizer
