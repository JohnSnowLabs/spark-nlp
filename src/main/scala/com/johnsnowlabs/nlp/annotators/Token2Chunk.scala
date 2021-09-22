/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/** Converts `TOKEN` type Annotations to `CHUNK` type.
 *
 * This can be useful if a entities have been already extracted as `TOKEN` and following annotators require `CHUNK` types.
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import com.johnsnowlabs.nlp.annotators.{Token2Chunk, Tokenizer}
 *
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
 * val token2chunk = new Token2Chunk()
 *   .setInputCols("token")
 *   .setOutputCol("chunk")
 *
 * val pipeline = new Pipeline().setStages(Array(
 *   documentAssembler,
 *   tokenizer,
 *   token2chunk
 * ))
 *
 * val data = Seq("One Two Three Four").toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.selectExpr("explode(chunk) as result").show(false)
 * +------------------------------------------+
 * |result                                    |
 * +------------------------------------------+
 * |[chunk, 0, 2, One, [sentence -> 0], []]   |
 * |[chunk, 4, 6, Two, [sentence -> 0], []]   |
 * |[chunk, 8, 12, Three, [sentence -> 0], []]|
 * |[chunk, 14, 17, Four, [sentence -> 0], []]|
 * +------------------------------------------+
 * }}}
 *
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
class Token2Chunk(override val uid: String) extends AnnotatorModel[Token2Chunk] with HasSimpleAnnotate[Token2Chunk] {


  /** Output Annotator Type : CHUNK
   *
   * @group anno
   * */
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** Input Annotator Type : TOKEN
   *
   * @group anno
   * */
  override val inputAnnotatorTypes: Array[String] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("TOKEN2CHUNK"))

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token =>
      Annotation(
        CHUNK,
        token.begin,
        token.end,
        token.result,
        token.metadata
      )
    }
  }

}

/**
 * This is the companion object of [[Token2Chunk]]. Please refer to that class for the documentation.
 */
object Token2Chunk extends DefaultParamsReadable[Token2Chunk]