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
package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.{Annotation, AnnotationImage}

private[johnsnowlabs] final case class BiEncoderEmbeddingPair(
    docEmbedding: Array[Float],
    imageEmbedding: Array[Float])
    extends Serializable

private[johnsnowlabs] trait BiEncoderMultimodal extends Serializable {

  def predict(
      documentAnnotations: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage]): Seq[BiEncoderEmbeddingPair]
}
