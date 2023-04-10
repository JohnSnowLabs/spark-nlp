/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.ml.crf.{FbCalculator, LinearChainCrfModel}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Annotated.{NerTaggedSentence, PosTaggedSentence}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util._
import org.apache.spark.sql.Dataset

import scala.collection.Map

/** Extracts Named Entities based on a CRF Model.
  *
  * This Named Entity recognition annotator allows for a generic model to be trained by utilizing
  * a CRF machine learning algorithm. The data should have columns of type `DOCUMENT, TOKEN, POS,
  * WORD_EMBEDDINGS`. These can be extracted with for example
  *   - a [[com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector SentenceDetector]],
  *   - a [[com.johnsnowlabs.nlp.annotators.Tokenizer Tokenizer]] and
  *   - a [[com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel PerceptronModel]].
  *
  * This is the instantiated model of the [[NerCrfApproach]]. For training your own model, please
  * see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val nerTagger = NerCrfModel.pretrained()
  *   .setInputCols("sentence", "token", "word_embeddings", "pos")
  *   .setOutputCol("ner"
  * }}}
  * The default model is `"ner_crf"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?task=Named+Entity+Recognition Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/crf-ner/ner_dl_crf.ipynb Examples]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  * import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
  * import org.apache.spark.ml.Pipeline
  *
  * // First extract the prerequisites for the NerCrfModel
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentence")
  *   .setOutputCol("token")
  *
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("word_embeddings")
  *
  * val posTagger = PerceptronModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("pos")
  *
  * // Then NER can be extracted
  * val nerTagger = NerCrfModel.pretrained()
  *   .setInputCols("sentence", "token", "word_embeddings", "pos")
  *   .setOutputCol("ner")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   embeddings,
  *   posTagger,
  *   nerTagger
  * ))
  *
  * val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("ner.result").show(false)
  * +------------------------------------+
  * |result                              |
  * +------------------------------------+
  * |[I-ORG, O, O, I-PER, O, O, I-LOC, O]|
  * +------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel NerDLModel]] for a deep learning based
  *   approach
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.ner.NerConverter NerConverter]] to further process the
  *   results
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class NerCrfModel(override val uid: String)
    extends AnnotatorModel[NerCrfModel]
    with HasSimpleAnnotate[NerCrfModel]
    with HasStorageRef {

  def this() = this(Identifiable.randomUID("NER"))

  /** List of Entities to recognize
    *
    * @group param
    */
  val entities = new StringArrayParam(this, "entities", "List of Entities to recognize")

  /** The CRF model
    *
    * @group param
    */
  val model: StructFeature[LinearChainCrfModel] =
    new StructFeature[LinearChainCrfModel](this, "crfModel")

  /** Additional dictionary to use as for features (Default: `Map.empty[String, String]`)
    *
    * @group param
    */
  val dictionaryFeatures: MapFeature[String, String] =
    new MapFeature[String, String](this, "dictionaryFeatures")

  /** Whether or not to calculate prediction confidence by token, included in metadata (Default:
    * `false`)
    *
    * @group param
    */
  val includeConfidence = new BooleanParam(
    this,
    "includeConfidence",
    "whether or not to calculate prediction confidence by token, includes in metadata")

  /** @group setParam */
  def setModel(crf: LinearChainCrfModel): NerCrfModel = set(model, crf)

  /** @group setParam */
  def setDictionaryFeatures(dictFeatures: DictionaryFeatures): this.type =
    set(dictionaryFeatures, dictFeatures.dict)

  /** @group setParam */
  def setEntities(toExtract: Array[String]): NerCrfModel = set(entities, toExtract)

  /** @group setParam */
  def setIncludeConfidence(c: Boolean): this.type = set(includeConfidence, c)

  /** @group getParam */
  def getIncludeConfidence: Boolean = $(includeConfidence)

  setDefault(dictionaryFeatures, () => Map.empty[String, String])
  setDefault(includeConfidence, false)

  /** Predicts Named Entities in input sentences
    *
    * @param sentences
    *   POS tagged and WordpieceEmbeddings sentences
    * @return
    *   sentences with recognized Named Entities
    */
  def tag(sentences: Seq[(PosTaggedSentence, WordpieceEmbeddingsSentence)])
      : Seq[NerTaggedSentence] = {
    require(model.isSet, "model must be set before tagging")

    val crf = $$(model)

    val fg = FeatureGenerator(new DictionaryFeatures($$(dictionaryFeatures)))
    sentences.map { case (sentence, withEmbeddings) =>
      val instance = fg.generate(sentence, withEmbeddings, crf.metadata)

      lazy val confidenceValues = {
        val fb = new FbCalculator(instance.items.length, crf.metadata)
        fb.calculate(instance, $$(model).weights, 1)
        fb.alpha
      }

      val labelIds = crf.predict(instance)

      val words = sentence.indexedTaggedWords
        .zip(labelIds.labels)
        .zipWithIndex
        .flatMap { case ((word, labelId), idx) =>
          val label = crf.metadata.labels(labelId)

          val alpha = if ($(includeConfidence)) {
            val scores = Some(confidenceValues.apply(idx))
            Some(
              crf.metadata.labels.zipWithIndex
                .filter(x => x._2 != 0)
                .map { case (t, i) =>
                  Map(
                    t -> scores
                      .getOrElse(Array.empty[String])
                      .lift(i)
                      .getOrElse(0.0f)
                      .toString)
                })
          } else None

          if (!isDefined(entities) || $(entities).isEmpty || $(entities).contains(label))
            Some(IndexedTaggedWord(word.word, label, word.begin, word.end, alpha))
          else
            None
        }

      TaggedSentence(words)
    }
  }

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    validateStorageRef(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)
    dataset
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sourceSentences = PosTagged.unpack(annotations)
    val withEmbeddings = WordpieceEmbeddingsSentence.unpack(annotations)
    val taggedSentences = tag(sourceSentences.zip(withEmbeddings))
    NerTagged.pack(taggedSentences)
  }

  def shrink(minW: Float): NerCrfModel = set(model, $$(model).shrink(minW))

  /** Input Annotator Types: DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS)

  /** Output Annotator Types: NAMED_ENTITY
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = NAMED_ENTITY

}

trait ReadablePretrainedNerCrf
    extends ParamsAndFeaturesReadable[NerCrfModel]
    with HasPretrained[NerCrfModel] {
  override val defaultModelName: Option[String] = Some("ner_crf")

  /** Java compliant-overrides */
  override def pretrained(): NerCrfModel = super.pretrained()

  override def pretrained(name: String): NerCrfModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): NerCrfModel = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): NerCrfModel =
    super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[NerCrfModel]]. Please refer to that class for the
  * documentation.
  */
object NerCrfModel extends ReadablePretrainedNerCrf
