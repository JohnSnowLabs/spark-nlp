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

package com.johnsnowlabs.nlp.annotators.parser.dep

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.common.Annotated.PosTaggedSentence
import com.johnsnowlabs.nlp.annotators.common.{
  DependencyParsed,
  DependencyParsedSentence,
  PosTagged
}
import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition._
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable

/** Unlabeled parser that finds a grammatical relation between two words in a sentence.
  *
  * Dependency parser provides information about word relationship. For example, dependency
  * parsing can tell you what the subjects and objects of a verb are, as well as which words are
  * modifying (describing) the subject. This can help you find precise answers to specific
  * questions.
  *
  * This is the instantiated model of the [[DependencyParserApproach]]. For training your own
  * model, please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val dependencyParserApproach = DependencyParserModel.pretrained()
  *   .setInputCols("sentence", "pos", "token")
  *   .setOutputCol("dependency")
  * }}}
  * The default model is `"dependency_conllu"`, if no name is provided. For available pretrained
  * models please see the [[https://nlp.johnsnowlabs.com/models Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/graph-extraction/graph_extraction_intro.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/parser/dep/DependencyParserApproachTestSpec.scala DependencyParserApproachTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
  * import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import org.apache.spark.ml.Pipeline
  *
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
  * val posTagger = PerceptronModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("pos")
  *
  * val dependencyParser = DependencyParserModel.pretrained()
  *   .setInputCols("sentence", "pos", "token")
  *   .setOutputCol("dependency")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   posTagger,
  *   dependencyParser
  * ))
  *
  * val data = Seq(
  *   "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
  *     "firm Federal Mogul."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(arrays_zip(token.result, dependency.result)) as cols")
  *   .selectExpr("cols['0'] as token", "cols['1'] as dependency").show(8, truncate = false)
  * +------------+------------+
  * |token       |dependency  |
  * +------------+------------+
  * |Unions      |ROOT        |
  * |representing|workers     |
  * |workers     |Unions      |
  * |at          |Turner      |
  * |Turner      |workers     |
  * |Newall      |say         |
  * |say         |Unions      |
  * |they        |disappointed|
  * +------------+------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel TypedDependencyParserMdoel]]
  *   to extract labels for the dependencies
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
class DependencyParserModel(override val uid: String)
    extends AnnotatorModel[DependencyParserModel]
    with HasSimpleAnnotate[DependencyParserModel] {
  def this() = this(Identifiable.randomUID(DEPENDENCY))

  /** Output annotation type : DEPENDENCY
    *
    * @group anno
    */
  override val outputAnnotatorType: String = DEPENDENCY

  /** Input annotation type : DOCUMENT, POS, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array[String](DOCUMENT, POS, TOKEN)

  /** Model that was set during training
    *
    * @group param
    */
  val perceptron: StructFeature[DependencyMaker] =
    new StructFeature[DependencyMaker](this, "perceptron")

  /** @group setParam */
  def setPerceptron(value: DependencyMaker): this.type = set(perceptron, value)

  /** @group getParam */
  def getDependencyParsedSentence(sentence: PosTaggedSentence): DependencyParsedSentence = {
    val dependencyParsedSentence = GreedyTransitionApproach.predict(sentence, $$(perceptron))
    dependencyParsedSentence
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val posTaggedSentences = PosTagged.unpack(annotations)
    val sentencesWithDependency = posTaggedSentences.map { sentence =>
      getDependencyParsedSentence(sentence)
    }
    val dependencyParser = DependencyParsed.pack(sentencesWithDependency)
    dependencyParser
  }
}

trait ReadablePretrainedDependency
    extends ParamsAndFeaturesReadable[DependencyParserModel]
    with HasPretrained[DependencyParserModel] {
  override val defaultModelName = Some("dependency_conllu")

  /** Java compliant-overrides */
  override def pretrained(): DependencyParserModel = super.pretrained()

  override def pretrained(name: String): DependencyParserModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): DependencyParserModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): DependencyParserModel =
    super.pretrained(name, lang, remoteLoc)
}

/** This is the companion object of [[DependencyParserModel]]. Please refer to that class for the
  * documentation.
  */
object DependencyParserModel extends ReadablePretrainedDependency
