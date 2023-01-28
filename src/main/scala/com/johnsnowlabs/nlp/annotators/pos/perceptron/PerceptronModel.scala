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

package com.johnsnowlabs.nlp.annotators.pos.perceptron

import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable

/** Averaged Perceptron model to tag words part-of-speech. Sets a POS tag to each word within a
  * sentence.
  *
  * This is the instantiated model of the
  * [[com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach PerceptronApproach]]. For
  * training your own model, please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val posTagger = PerceptronModel.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("pos")
  * }}}
  * The default model is `"pos_anc"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Part+of+Speech+Tagging Models Hub]]. Additionally,
  * pretrained pipelines are available for this module, see
  * [[https://nlp.johnsnowlabs.com/docs/en/pipelines Pipelines]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/training/french/Train-Perceptron-French.ipynb Examples]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val posTagger = PerceptronModel.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("pos")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   posTagger
  * ))
  *
  * val data = Seq("Peter Pipers employees are picking pecks of pickled peppers").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(pos) as pos").show(false)
  * +-------------------------------------------+
  * |pos                                        |
  * +-------------------------------------------+
  * |[pos, 0, 4, NNP, [word -> Peter], []]      |
  * |[pos, 6, 11, NNP, [word -> Pipers], []]    |
  * |[pos, 13, 21, NNS, [word -> employees], []]|
  * |[pos, 23, 25, VBP, [word -> are], []]      |
  * |[pos, 27, 33, VBG, [word -> picking], []]  |
  * |[pos, 35, 39, NNS, [word -> pecks], []]    |
  * |[pos, 41, 42, IN, [word -> of], []]        |
  * |[pos, 44, 50, JJ, [word -> pickled], []]   |
  * |[pos, 52, 58, NNS, [word -> peppers], []]  |
  * +-------------------------------------------+
  * }}}
  *
  * @param uid
  *   Internal constructor requirement for serialization of params
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
class PerceptronModel(override val uid: String)
    extends AnnotatorModel[PerceptronModel]
    with HasSimpleAnnotate[PerceptronModel]
    with PerceptronPredictionUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** POS model
    *
    * @group param
    */
  val model: StructFeature[AveragedPerceptron] =
    new StructFeature[AveragedPerceptron](this, "POS Model")

  /** Output annotator types : POS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = POS

  /** Input annotator types : TOKEN, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  def this() = this(Identifiable.randomUID("POS"))

  /** @group getParam */
  def getModel: AveragedPerceptron = $$(model)

  /** @group setParam */
  def setModel(targetModel: AveragedPerceptron): this.type = set(model, targetModel)

  /** One to one annotation standing from the Tokens perspective, to give each word a
    * corresponding Tag
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)
    val tagged = tag($$(model), tokenizedSentences.toArray)
    PosTagged.pack(tagged)
  }
}

trait ReadablePretrainedPerceptron
    extends ParamsAndFeaturesReadable[PerceptronModel]
    with HasPretrained[PerceptronModel] {
  override val defaultModelName = Some("pos_anc")

  /** Java compliant-overrides */
  override def pretrained(): PerceptronModel = super.pretrained()

  override def pretrained(name: String): PerceptronModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): PerceptronModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): PerceptronModel =
    super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[PerceptronModel]]. Please refer to that class for the
  * documentation.
  */
object PerceptronModel extends ReadablePretrainedPerceptron
