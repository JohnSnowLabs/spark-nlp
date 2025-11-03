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

import com.johnsnowlabs.nlp.LegacyMetadataSupport.ParamsReflection
import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.{compact, parse, render}
import org.json4s.{DefaultFormats, JNothing, JNull, JObject, JValue}

import scala.collection.mutable.ArrayBuffer
import scala.util.{Failure, Success, Try}

class FeaturesReader[T <: HasFeatures](
    baseReader: MLReader[T],
    onRead: (T, String, SparkSession) => Unit)
    extends MLReader[T] {

  override def load(path: String): T = {

    val instance =
      try {
        // Let Spark's own loader handle modern bundles.
        baseReader.load(path)
      } catch {
        case e: NoSuchElementException if isMissingParamError(e) =>
          // Reconstruct legacy models that referenced params removed in newer releases.
          loadWithLegacyParams(path)
      }

    for (feature <- instance.features) {
      val value = feature.deserialize(sparkSession, path, feature.name)
      feature.setValue(value)
    }

    onRead(instance, path, sparkSession)

    instance
  }

  private def isMissingParamError(e: NoSuchElementException): Boolean = {
    val msg = Option(e.getMessage).getOrElse("")
    msg.contains("Param")
  }

  private def loadWithLegacyParams(path: String): T = {
    val metadata = LegacyMetadataSupport.load(path, sparkSession)
    val cls = Class.forName(metadata.className)
    val ctor = cls.getConstructor(classOf[String])
    val instance = ctor.newInstance(metadata.uid).asInstanceOf[Params]
    setParamsIgnoringUnknown(instance, metadata)
    instance.asInstanceOf[T]
  }

  private def setParamsIgnoringUnknown(
      instance: Params,
      metadata: LegacyMetadataSupport.Metadata): Unit = {
    // Replay active params; skip mismatches so legacy bundles still come back.
    assignParams(instance, metadata.params, isDefault = false, metadata)

    val hasDefaultSection = metadata.defaultParams != JNothing && metadata.defaultParams != JNull
    if (hasDefaultSection) {
      // If the metadata carried defaults, restore only those that still exists.
      assignParams(instance, metadata.defaultParams, isDefault = true, metadata)
    }
  }

  private def assignParams(
      instance: Params,
      jsonParams: JValue,
      isDefault: Boolean,
      metadata: LegacyMetadataSupport.Metadata): Unit = {
    jsonParams match {
      case JObject(pairs) =>
        pairs.foreach { case (paramName, jsonValue) =>
          if (instance.hasParam(paramName)) {
            val param = instance.getParam(paramName)
            val value = param.jsonDecode(compact(render(jsonValue)))
            if (isDefault) {
              // Spark keeps setDefault protected; call it via reflection to restore legacy defaults.
              ParamsReflection.setDefault(instance, param, value)
            } else {
              instance.set(param, value)
            }
          }
        }
      case JNothing | JNull =>
      case other =>
        throw new IllegalArgumentException(
          s"Cannot recognize JSON metadata when loading legacy params for ${metadata.className}: $other")
    }
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
      // Read params, features and model via FeaturesReader.load
      baseReader.load(path)
    } match {
      case Success(value) => value
      case Failure(_: java.util.NoSuchElementException) =>
        println(
          s"Failed to load all parameters from $path, attempting fallback loader. " +
            s"Parameters will be set to default values.")
        fallbackLoad(path, sparkSession)
      case Failure(_: java.lang.ClassCastException) =>
        println(
          s"Failed to cast to class of $path, attempting fallback loader. " +
            s"Parameters will be set to default values.")
        fallbackLoad(path, sparkSession)
      case Failure(exception) => throw exception
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

// Minimal metadata parser + helper utilities for replaying legacy params.
protected object LegacyMetadataSupport {

  object ParamsReflection {
    private val setDefaultMethod = {
      val maybeMethod = classOf[Params].getDeclaredMethods.find { method =>
        method.getName == "setDefault" && method.getParameterCount == 2
      }

      maybeMethod match {
        case Some(method) =>
          method.setAccessible(true)
          method
        case None =>
          throw new NoSuchMethodException("Params.setDefault(Param, value) not found via reflection")
      }
    }

    def setDefault[T](
        params: Params,
        param: org.apache.spark.ml.param.Param[T],
        value: T): Unit = {
      setDefaultMethod.invoke(params, param, toAnyRef(value))
    }

    // Mirror JVM boxing rules so reflection can call the protected method safely.
    private def toAnyRef(value: Any): AnyRef = {
      if (value == null) {
        null
      } else {
        value match {
          case v: AnyRef => v
          case v: Boolean => java.lang.Boolean.valueOf(v)
          case v: Byte => java.lang.Byte.valueOf(v)
          case v: Short => java.lang.Short.valueOf(v)
          case v: Int => java.lang.Integer.valueOf(v)
          case v: Long => java.lang.Long.valueOf(v)
          case v: Float => java.lang.Float.valueOf(v)
          case v: Double => java.lang.Double.valueOf(v)
          case v: Char => java.lang.Character.valueOf(v)
          case other =>
            throw new IllegalArgumentException(
              s"Unsupported default value type ${other.getClass}")
        }
      }
    }
  }

  case class Metadata(
      className: String,
      uid: String,
      sparkVersion: String,
      params: JValue,
      defaultParams: JValue,
      metadataJson: String)

  def load(path: String, spark: SparkSession): Metadata = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataStr = spark.sparkContext.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr)
  }

  private def parseMetadata(metadataStr: String): Metadata = {
    val metadata = parse(metadataStr)
    implicit val format: DefaultFormats.type = DefaultFormats

    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val sparkVersion = (metadata \ "sparkVersion").extractOpt[String].getOrElse("0.0")
    val params = metadata \ "paramMap"
    val defaultParams = metadata \ "defaultParamMap"

    Metadata(className, uid, sparkVersion, params, defaultParams, metadataStr)
  }
}
