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

package com.johnsnowlabs.nlp.annotators.ner

import org.apache.spark.ml.param.{IntParam, Param, Params, StringArrayParam}

/** @groupname anno Annotator types
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
trait NerApproach[T <: NerApproach[_]] extends Params {

  /** Column with label per each token
    *
    * @group param
    */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each token")

  /** Entities to recognize
    *
    * @group param
    */
  val entities = new StringArrayParam(this, "entities", "Entities to recognize")

  /** Minimum number of epochs to train
    *
    * @group param
    */
  val minEpochs = new IntParam(this, "minEpochs", "Minimum number of epochs to train")

  /** Maximum number of epochs to train
    *
    * @group param
    */
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")

  /** Random seed
    *
    * @group param
    */
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")

  /** Column with label per each token
    *
    * @group setParam
    */
  def setLabelColumn(column: String): T = set(labelColumn, column).asInstanceOf[T]

  /** Entities to recognize
    *
    * @group setParam
    */
  def setEntities(tags: Array[String]): T = set(entities, tags).asInstanceOf[T]

  /** Minimum number of epochs to train
    *
    * @group setParam
    */
  def setMinEpochs(epochs: Int): T = set(minEpochs, epochs).asInstanceOf[T]

  /** Maximum number of epochs to train
    *
    * @group setParam
    */
  def setMaxEpochs(epochs: Int): T = set(maxEpochs, epochs).asInstanceOf[T]

  /** Random seed
    *
    * @group setParam
    */
  def setRandomSeed(seed: Int): T = set(randomSeed, seed).asInstanceOf[T]

  /** Minimum number of epochs to train
    *
    * @group getParam
    */
  def getMinEpochs: Int = $(minEpochs)

  /** Maximum number of epochs to train
    *
    * @group getParam
    */
  def getMaxEpochs: Int = $(maxEpochs)

  /** Random seed
    *
    * @group getParam
    */
  def getRandomSeed: Int = $(randomSeed)

}

object Verbose extends Enumeration {
  type Level = Value

  val All: Verbose.Value = Value(0)
  val PerStep: Verbose.Value = Value(1)
  val Epochs: Verbose.Value = Value(2)
  val TrainingStat: Verbose.Value = Value(3)
  val Silent: Verbose.Value = Value(4)
}

object ModelMetrics {

  val microF1 = "f1_micro"
  val macroF1 = "f1_macro"
  val testMicroF1 = "test_micro_f1"
  val testMacroF1 = "test_macro_f1"
  val valMicroF1 = "val_micro_f1"
  val valMacroF1 = "val_macro_f1"
  val loss = "loss"

}
