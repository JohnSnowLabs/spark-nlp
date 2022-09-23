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

object AnnotatorType {
  val AUDIO = "audio"
  val DOCUMENT = "document"
  val IMAGE = "image"
  val TOKEN = "token"
  val WORDPIECE = "wordpiece"
  val WORD_EMBEDDINGS = "word_embeddings"
  val SENTENCE_EMBEDDINGS = "sentence_embeddings"
  val CATEGORY = "category"
  val DATE = "date"
  val ENTITY = "entity"
  val SENTIMENT = "sentiment"
  val POS = "pos"
  val CHUNK = "chunk"
  val NAMED_ENTITY = "named_entity"
  val NEGEX = "negex"
  val DEPENDENCY = "dependency"
  val LABELED_DEPENDENCY = "labeled_dependency"
  val LANGUAGE = "language"
  val NODE = "node"
  val TABLE = "table"
  val DUMMY = "dummy"

}
