/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.common

/**
  * Structure to hold Sentences as list of words and POS-tags
  * @param taggedWords Word tag pairs
  */
case class TaggedSentence(taggedWords: Array[TaggedWord], indexedTaggedWords: Array[IndexedTaggedWord] = Array()) {
  def this(indexedTaggedWords: Array[IndexedTaggedWord]) = this(indexedTaggedWords.map(_.toTaggedWord), indexedTaggedWords)
  /** Recurrently needed to access all words */
  val words: Array[String] = taggedWords.map(_.word)
  /** Recurrently needed to access all tags */
  val tags: Array[String] = taggedWords.map(_.tag)
  /** ready function to return pairwise tagged words */
  def tupleWords: Array[(String, String)] = words.zip(tags)
  def mapWords: Map[String, String] = tupleWords.toMap
}

object TaggedSentence {
  def apply(indexedTaggedWords: Array[IndexedTaggedWord]) = new TaggedSentence(indexedTaggedWords.map(_.toTaggedWord), indexedTaggedWords)
}