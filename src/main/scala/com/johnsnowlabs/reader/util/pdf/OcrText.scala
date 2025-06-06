/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.reader.util.pdf

import com.johnsnowlabs.nlp.IAnnotation

case class OcrText(
    text: String,
    metadata: Map[String, String],
    content: Array[Byte] = Array.empty[Byte])
    extends IAnnotation {

  override def annotatorType: String = "image_to_text"
  def begin = 0
  def end = text.length
  def result = text

}
