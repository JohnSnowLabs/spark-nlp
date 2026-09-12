/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.ml.onnx

import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtException.OrtErrorCode
import org.slf4j.LoggerFactory
import java.util.regex.Pattern
import scala.util.control.NonFatal

/** Failure-first, one-retry state machine for ONNX Runtime CUDA provider registration. */
private[onnx] object CudaProviderRecovery {

  private val logger = LoggerFactory.getLogger("CudaProviderRecovery")

  private val recoverableCodes = Set(OrtErrorCode.ORT_FAIL, OrtErrorCode.ORT_EP_FAIL)
  private val providerLoadAnchor = "failed to load library libonnxruntime_providers_cuda.so"

  def configure[T <: AutoCloseable](
      mode: => NativeLibraryPreloader.Mode,
      createOptions: () => T,
      addCudaProvider: T => Unit,
      preload: () => Unit): T = {
    val firstOptions = createOptions()
    try {
      addCudaProvider(firstOptions)
      firstOptions
    } catch {
      case original: Throwable =>
        closeAfterFailure(firstOptions, original)
        if (!isRecoverable(original)) throw original
        logger.warn(
          "ONNX CUDA provider registration failed because a native dependency was unavailable; evaluating recovery configuration")
        val recoveryMode =
          try mode
          catch {
            case configurationFailure: Throwable =>
              configurationFailure.addSuppressed(original)
              throw configurationFailure
          }
        if (recoveryMode == NativeLibraryPreloader.Off) throw original

        try preload()
        catch {
          case preloadFailure: Throwable =>
            preloadFailure.addSuppressed(original)
            throw preloadFailure
        }
        val retryOptions =
          try createOptions()
          catch {
            case creationFailure: Throwable =>
              creationFailure.addSuppressed(original)
              throw creationFailure
          }
        try {
          addCudaProvider(retryOptions)
          logger.info(
            "ONNX CUDA provider registration retry succeeded after native dependency preload")
          retryOptions
        } catch {
          case retryFailure: Throwable =>
            closeAfterFailure(retryOptions, retryFailure)
            retryFailure.addSuppressed(original)
            throw retryFailure
        }
    }
  }

  def isRecoverable(error: Throwable): Boolean =
    isRecoverable(error, () => NativeLibraryPreloader.packagedDependencies)

  private[onnx] def isRecoverable(error: Throwable, dependencies: () => Seq[String]): Boolean =
    try {
      error match {
        case ortError: OrtException if recoverableCodes.contains(ortError.getCode) =>
          val message = throwableMessages(ortError).mkString(" ").toLowerCase
          val namesExactMissingDependency = dependencies().exists { library =>
            val missingMarker =
              "(?i)(?:^|[^A-Za-z0-9_.-])" + Pattern.quote(library) +
                ":\\s*(?:cannot open shared object file|no such file or directory|error loading shared library)"
            Pattern.compile(missingMarker).matcher(message).find()
          }
          message.contains(providerLoadAnchor) && namesExactMissingDependency
        case _ => false
      }
    } catch {
      case NonFatal(_) => false
    }

  private val MaxDiagnosticMessageChars = 4096

  private def throwableMessages(error: Throwable): Seq[String] = {
    val messages = Seq.newBuilder[String]
    var current = error
    var depth = 0
    while (current != null && depth < 8) {
      messages += Option(current.getMessage).getOrElse("").take(MaxDiagnosticMessageChars)
      current = current.getCause
      depth += 1
    }
    messages.result()
  }

  private def closeAfterFailure(resource: AutoCloseable, failure: Throwable): Unit = {
    if (resource == null) return
    try resource.close()
    catch {
      case closeFailure: Throwable => failure.addSuppressed(closeFailure)
    }
  }
}
