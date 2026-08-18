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

import com.johnsnowlabs.util.ConfigHelper
import org.apache.spark.internal.Logging
import org.apache.spark.ml.util.{DefaultParamsReadable, MLReader}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal
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

private[nlp] object FallbackLoaderLogging {

  sealed trait Mode
  case object Off extends Mode
  case object Summary extends Mode
  case object Full extends Mode

  final case class ParsedMode(mode: Mode, invalidValue: Option[String] = None)

  val MaxCauseMessageLength: Int = 200

  def parseMode(configuredValue: Option[String]): ParsedMode = configuredValue match {
    case None => ParsedMode(Off)
    case Some(value) =>
      value.trim.toLowerCase(java.util.Locale.ROOT) match {
        case "off" => ParsedMode(Off)
        case "summary" => ParsedMode(Summary)
        case "full" => ParsedMode(Full)
        case _ => ParsedMode(Off, Some(value))
      }
  }

  def summary(modelType: String, throwable: Throwable): String = {
    val modelDescription = Option(modelType)
      .map(normalizeWhitespace)
      .filter(_.nonEmpty)
      .map(value => s" for $value")
      .getOrElse("")
    val causeDescription = deepestMeaningfulCause(throwable)

    s"Spark NLP fallback loader activated$modelDescription. Cause: $causeDescription. " +
      s"Set ${ConfigHelper.fallbackLoaderLogMode}=full for the complete stack trace."
  }

  def invalidModeWarning(value: String): String = {
    val normalizedValue = normalizeWhitespace(value) match {
      case "" => "<empty>"
      case other => truncate(other)
    }

    s"Unsupported value '$normalizedValue' for ${ConfigHelper.fallbackLoaderLogMode}; " +
      "valid options are off, summary, and full. Treating it as off."
  }

  private def deepestMeaningfulCause(throwable: Throwable): String = {
    val causes = ArrayBuffer.empty[Throwable]
    val visited = scala.collection.mutable.Set.empty[Throwable]
    var current = throwable

    while (current != null && !visited.contains(current)) {
      causes.append(current)
      visited.add(current)
      current = current.getCause
    }

    val deepest = causes.lastOption.getOrElse(throwable)
    causes.reverseIterator
      .map(cause => (cause, Option(cause.getMessage).map(sanitizeExceptionMessage).getOrElse("")))
      .find(_._2.nonEmpty)
      .map { case (cause, message) => s"${exceptionClassName(cause)}: $message" }
      .getOrElse(exceptionClassName(deepest))
  }

  private def exceptionClassName(throwable: Throwable): String = {
    val simpleName = throwable.getClass.getSimpleName
    if (simpleName.nonEmpty) simpleName else throwable.getClass.getName
  }

  private def normalizeWhitespace(value: String): String = value.replaceAll("\\s+", " ").trim

  private def sanitizeExceptionMessage(value: String): String = {
    val normalized = normalizeWhitespace(value)
    val stackTraceIndex =
      normalized.toLowerCase(java.util.Locale.ROOT).indexOf("driver stacktrace")
    val withoutDriverStackTrace =
      if (stackTraceIndex >= 0) normalized.take(stackTraceIndex).trim else normalized
    truncate(withoutDriverStackTrace)
  }

  private def truncate(value: String): String =
    value.take(MaxCauseMessageLength)
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
    extends MLReader[T]
    with Logging {

  protected[nlp] def modelType: String = ""

  protected[nlp] def warn(message: String): Unit = logWarning(message)

  protected[nlp] def warn(message: String, throwable: Throwable): Unit =
    logWarning(message, throwable)

  private def logFallbackFailure(throwable: Throwable): Unit = {
    val parsedMode = FallbackLoaderLogging.parseMode(
      sparkSession.conf.getOption(ConfigHelper.fallbackLoaderLogMode))

    parsedMode.invalidValue.foreach(value =>
      warn(FallbackLoaderLogging.invalidModeWarning(value)))

    parsedMode.mode match {
      case FallbackLoaderLogging.Off =>
      case FallbackLoaderLogging.Summary =>
        warn(FallbackLoaderLogging.summary(modelType, throwable))
      case FallbackLoaderLogging.Full =>
        warn(FallbackLoaderLogging.summary(modelType, throwable), throwable)
    }
  }

  override def load(path: String): T = {
    Try {
      // Read params, features and model via FeaturesReader.load
      baseReader.load(path)
    } match {
      case Success(value) => value
      case Failure(e: Throwable) =>
        try logFallbackFailure(e)
        catch {
          case NonFatal(_) =>
        }
        fallbackLoad(path, sparkSession)
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

  override def read: MLReader[T] = {
    val readableModelType = getClass.getSimpleName.stripSuffix("$")
    new FeaturesFallbackReader(super.read, onRead, fallbackLoad) {
      override protected[nlp] def modelType: String = readableModelType
    }
  }
}
