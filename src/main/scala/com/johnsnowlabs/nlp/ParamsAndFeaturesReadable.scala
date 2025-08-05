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

package com.johnsnowlabs.nlp

import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Try}

class FeaturesReader[T <: HasFeatures](
    baseReader: MLReader[T],
    onRead: (T, String, SparkSession) => Unit)
    extends MLReader[T] {

  override def load(path: String): T = {

    val instance = baseReader.load(path)

    for (feature <- instance.features) {
      val value = feature.deserialize(sparkSession, path, feature.name)
      feature.setValue(value)
    }

    onRead(instance, path, sparkSession)

    instance
  }
}

trait ParamsAndFeaturesReadable[T <: HasFeatures] extends DefaultParamsReadable[T] {

  protected val readers: ArrayBuffer[(T, String, SparkSession) => Unit] =
    ArrayBuffer.empty[(T, String, SparkSession) => Unit]

  protected def onRead(instance: T, path: String, session: SparkSession): Unit = {
    for (reader <- readers) {
      reader(instance, path, session)
    }
  }

  def addReader(reader: (T, String, SparkSession) => Unit): Unit = {
    readers.append(reader)
  }

  override def read: MLReader[T] =
    new FeaturesReader(
      super.read,
      (instance: T, path: String, spark: SparkSession) => onRead(instance, path, spark))
}

/** MLReader that loads a model with params and features, and has a fallback mechanism.
  *
  * The fallback load will be called in case there is an exception during Spark loading (i.e.
  * missing parameters or features).
  *
  * Usually, you might want to call `loadSavedModel` in the `fallbackLoad` method to load a model
  * with default params.
  *
  * @param baseReader
  *   The default spark reader
  * @param onRead
  *   A function that will be called after the model is loaded, allowing to set a model
  * @param fallbackLoad
  *   A fallback function that will be called in case the main reader fails to load
  * @tparam T
  *   The type of the model that extends HasFeatures
  */
class FeaturesFallbackReader[T <: HasFeatures](
    baseReader: MLReader[T],
    onRead: (T, String, SparkSession) => Unit,
    fallbackLoad: (String, SparkSession) => T = null)
    extends MLReader[T] {

  override def load(path: String): T = {
    Try {
      // Read params, features and model
      val instance = baseReader.load(path)
      loadFeatures(path, instance)
      onRead(instance, path, sparkSession)

      instance
    } match {
      case Failure(_) =>
        // TODO: Logger warn instead?
        println(
          s"Failed to load all parameters from $path, attempting fallback loader. " +
            s"Parameters will be set to default values.")
        fallbackLoad(path, sparkSession)
      case Success(value) => value
    }
  }

  private def loadFeatures(path: String, instance: T): Unit = {
    for (feature <- instance.features) {
      val value = feature.deserialize(sparkSession, path, feature.name)
      feature.setValue(value)
    }
  }
}

/** Enables loading models with params and features with a fallback mechanism. The `fallbackLoad`
  * function will be called in case there is an exception during Spark loading (i.e. missing
  * parameters or features).
  *
  * Usually, you might want to call `loadSavedModel` in the `fallbackLoad` method to load a model
  * with default params.
  *
  * @tparam T
  *   The type of the model that extends HasFeatures
  */
trait ParamsAndFeaturesFallbackReadable[T <: HasFeatures] extends ParamsAndFeaturesReadable[T] {

  /** Fallback loader for when the main reader fails to load the model (e.g., missing
    * params/features).
    *
    * For example, we could use loadSavedModel to load a model with default parameters and
    * features (if the model in the folder supports it).
    *
    * @param folder
    *   the folder where the model is stored
    * @param spark
    *   the Spark session
    * @return
    *   an instance of the model with default parameters and features loaded
    */
  def fallbackLoad(folder: String, spark: SparkSession): T

  override def read: MLReader[T] = new FeaturesFallbackReader(super.read, onRead, fallbackLoad)
}
