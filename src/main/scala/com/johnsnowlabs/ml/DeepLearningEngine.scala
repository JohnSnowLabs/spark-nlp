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

package com.johnsnowlabs.ml

import com.johnsnowlabs.ml.pytorch.{PytorchWrapper, WritePytorchModel}
import com.johnsnowlabs.ml.tensorflow.{TensorflowParams, TensorflowWrapper}
import com.johnsnowlabs.nlp.embeddings.TransformerEmbeddings
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Model
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.SparkSession

trait DeepLearningEngine[T <: TransformerEmbeddings, P <: TransformerEmbeddings, E <: Model[E]]
  extends TensorflowParams with WritePytorchModel {

  val deepLearningEngine = new Param[String](this, "deepLearningEngine",
    "Deep Learning engine for creating embeddings [tensorflow|pytorch]")

  def getDeepLearningEngine: String = $(deepLearningEngine).toLowerCase

  def setDeepLearningEngine(value: String): this.type = {
    set(deepLearningEngine, value)
  }

  setDefault(deepLearningEngine -> "tensorflow")

  protected var tfModel: Option[Broadcast[T]] = None

  protected var pytorchModel: Option[Broadcast[P]] = None

  private val unmatchDeepLearningEngineError = "Please verify that deep learning engine parameter matches your model."

  /** @group getParam */
  def getModelIfNotSet: T = {
    if (tfModel.isEmpty) {
      throw new NoSuchElementException(s"Tensorflow model is empty. $unmatchDeepLearningEngineError")
    }
    tfModel.get.value
  }

  /** @group getParam */
  def getPytorchModelIfNotSet: P = {
    if (pytorchModel.isEmpty) {
      throw new NoSuchElementException(s"Pytorch model is empty. $unmatchDeepLearningEngineError")
    }
    pytorchModel.get.value
  }

  def setModelIfNotSet(spark: SparkSession, tensorflowWrapper: TensorflowWrapper): E

  def setPytorchModelIfNotSet(spark: SparkSession, pytorchWrapper: PytorchWrapper): E

}
