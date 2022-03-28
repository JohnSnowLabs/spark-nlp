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

package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

import com.johnsnowlabs.nlp.annotators.common.{Sentence, SentenceSplit}
import com.johnsnowlabs.nlp.annotators.sbd.SentenceDetectorParams
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate, JavaAnnotation}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.JavaConverters._

/**
  * Annotator that detects sentence boundaries using any provided approach.
  *
  * Each extracted sentence can be returned in an Array or exploded to separate rows,
  * if `explodeSentences` is set to `true`.
  *
  * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb Spark NLP Workshop]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
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
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence
  * ))
  *
  * val data = Seq("This is my first sentence. This my second. How about a third?").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(sentence) as sentences").show(false)
  * +------------------------------------------------------------------+
  * |sentences                                                         |
  * +------------------------------------------------------------------+
  * |[document, 0, 25, This is my first sentence., [sentence -> 0], []]|
  * |[document, 27, 41, This my second., [sentence -> 1], []]          |
  * |[document, 43, 60, How about a third?, [sentence -> 2], []]       |
  * +------------------------------------------------------------------+
  * }}}
  *
  * @see [[com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel SentenceDetectorDLModel]] for pretrained models
  * @param uid internal constructor requirement for serialization of params
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
class SentenceDetector(override val uid: String) extends AnnotatorModel[SentenceDetector] with HasSimpleAnnotate[SentenceDetector] with SentenceDetectorParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  def this() = this(Identifiable.randomUID("SENTENCE"))

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = DOCUMENT
  /** Input annotator type : DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  lazy val model: PragmaticMethod =
    if ($(customBounds).nonEmpty && $(useCustomBoundsOnly))
      new CustomPragmaticMethod($(customBounds))
    else if ($(customBounds).nonEmpty)
      new MixedPragmaticMethod($(useAbbrevations), $(detectLists), $(customBounds))
    else
      new DefaultPragmaticMethod($(useAbbrevations), $(detectLists))

  def tag(document: String): Array[Sentence] = {
    model.extractBounds(
      document
    ).flatMap(sentence => {
      var currentStart = sentence.start
      get(splitLength).map(splitLength => truncateSentence(sentence.content, splitLength)).getOrElse(Array(sentence.content))
        .zipWithIndex.map{ case (truncatedSentence, index) =>
        val currentEnd = currentStart + truncatedSentence.length - 1
        val result = Sentence(truncatedSentence, currentStart, currentEnd, index)
        /** +1 because of shifting to the next token begin. +1 because of a whitespace jump to next token. */
        currentStart = currentEnd + 2
        result
      }
    })
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    /** Preload model */
    model

    dataset
  }

  /**
    * Uses the model interface to prepare the context and extract the boundaries
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return One to many annotation relationship depending on how many sentences there are in the document
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val docs = annotations.map(_.result)
    val sentences = docs.flatMap(doc => tag(doc))
      .filter(t => t.content.nonEmpty && t.content.length >= $(minLength) && get(maxLength).forall(m => t.content.length <= m))
    SentenceSplit.pack(sentences)
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    import org.apache.spark.sql.functions.{array, col, explode}
    if ($(explodeSentences)) {
      dataset
        .select(dataset.columns.filterNot(_ == getOutputCol).map(col) :+ explode(col(getOutputCol)).as("_tmp"):_*)
        .withColumn(getOutputCol, array(col("_tmp")).as(getOutputCol, dataset.schema.fields.find(_.name == getOutputCol).get.metadata))
        .drop("_tmp")
    }
    else dataset
  }

}

/**
 * This is the companion object of [[SentenceDetector]]. Please refer to that class for the documentation.
 */
object SentenceDetector extends DefaultParamsReadable[SentenceDetector]