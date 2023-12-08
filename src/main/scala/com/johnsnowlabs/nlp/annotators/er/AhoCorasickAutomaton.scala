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
package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import com.johnsnowlabs.nlp.annotators.common.Sentence

import scala.collection.mutable.ArrayBuffer

/** Aho-Corasick Algorithm: https://dl.acm.org/doi/10.1145/360825.360855 A simple, efficient
  * algorithm to locate all occurrences of any of a finite number of keywords in a string of text.
  * The algorithm consists of constructing a finite state pattern matching machine from the
  * keywords and then using the pattern matching machine to process the text string in a single
  * pass. The complexity of constructing a pattern matching machine and searching the text is
  * linear to the total length of given patterns and the length of a text, respectively.
  */
class AhoCorasickAutomaton(
    var alphabet: String,
    patterns: Array[EntityPattern],
    caseSensitive: Boolean = false)
    extends Serializable {

  alphabet = if (alphabet.contains(" ")) alphabet else alphabet + " "
  private val flattenEntityPatterns: Array[FlattenEntityPattern] = patterns.flatMap {
    entityPattern =>
      entityPattern.patterns.map { pattern =>
        val keyword = if (caseSensitive) pattern else pattern.toLowerCase
        FlattenEntityPattern(entityPattern.label, keyword, entityPattern.id)
      }
  }

  private val ALPHABET_SIZE = alphabet.length
  private val MAX_NODES = flattenEntityPatterns.map(value => value.keyword.length).sum + 1

  var nodes: Array[Option[Node]] = Array.fill(MAX_NODES)(None)
  var nodeCount: Int = 1

  class Node extends Serializable {

    var parentState: Int = -1
    var charFromParent: Option[Char] = None
    var suffixLink: Int = -1
    var children: Array[Int] = Array.fill(ALPHABET_SIZE)(-1)
    var transitions: Array[Int] =
      Array.fill(ALPHABET_SIZE)(-1) // Transition Table aka Goto function
    var isLeaf: Boolean = false
    var entity: String = ""
    var id: String = ""
  }

  private val edges: Map[Char, Int] = alphabet.toCharArray.zipWithIndex.map {
    case (char, index) => (char, index)
  }.toMap

  initializeTrie()

  private def initializeTrie(): Unit = {
    nodes(0) = Some(new Node())
    nodes(0).get.suffixLink = 0
    nodes(0).get.parentState = -1

    flattenEntityPatterns.foreach(pattern => addPattern(pattern))
  }

  /** First step of Aho-Corasick algorithm: Build a Finite State Automaton as a keyword trie in
    * which the nodes represent the state and the edges between nodes are labeled by characters
    * that cause the transitions between nodes. The trie is an efficient implementation of a
    * dictionary of strings.
    */
  private def addPattern(pattern: FlattenEntityPattern): Unit = {
    var state = 0
    pattern.keyword.toCharArray.foreach { char =>
      val edgeIndex = edges.getOrElse(
        char, {
          val errorMessage = getAlphabetErrorMessage(char)
          throw new UnsupportedOperationException(errorMessage)
        })

      if (nodes(state).get.children(edgeIndex) == -1) {
        nodes(nodeCount) = Some(new Node())
        nodes(nodeCount).get.parentState = state
        nodes(nodeCount).get.charFromParent = Some(char)
        nodes(state).get.children(edgeIndex) = nodeCount
        nodeCount = nodeCount + 1
      }
      state = nodes(state).get.children(edgeIndex)
    }
    nodes(state).get.isLeaf = true
    nodes(state).get.entity = pattern.entity
    if (pattern.id.isDefined) nodes(state).get.id = pattern.id.get
  }

  /** Second step of Aho-Corasick algorithm: The algorithm starts at the input text’s beginning
    * and in the root state during searching for patterns. It processes the input string in a
    * single pass, and all occurrences of keywords are found, even if they overlap each other.
    */
  def searchPatternsInText(sentence: Sentence): Seq[Annotation] = {
    var previousState = 0
    val chunk: ArrayBuffer[(Char, Int)] = ArrayBuffer.empty
    val chunkAnnotations: ArrayBuffer[Annotation] = ArrayBuffer.empty

    sentence.content.zipWithIndex.foreach { case (char, index) =>
      val currentChar = if (caseSensitive) char else char.toLower
      val state = findNextState(previousState, currentChar)

      if (state > 0) {
        chunk.append((char, index))
      }

      if (state == 0 && previousState > 0) {
        val node = nodes(previousState).get
        if (node.isLeaf && node.entity.nonEmpty) {
          val chunkAnnotation = buildAnnotation(chunk, node.entity, node.id, sentence)
          chunkAnnotations.append(chunkAnnotation)
          chunk.clear()
        } else chunk.clear()
      }

      previousState = state
    }

    if (chunk.nonEmpty) {
      val node = nodes(previousState).get
      if (node.entity.nonEmpty) {
        val chunkAnnotation = buildAnnotation(chunk, node.entity, node.id, sentence)
        chunkAnnotations.append(chunkAnnotation)
      }
      chunk.clear()
    }

    chunkAnnotations
  }

  private def findNextState(state: Int, char: Char): Int = {

    val newLine = System.getProperty("line.separator")
    if (newLine == char.toString) return 0

    val edgeIndex: Int = edges.getOrElse(char, -1)
    if (edgeIndex == -1) {
      val errorMessage = getAlphabetErrorMessage(char)
      throw new UnsupportedOperationException(errorMessage)
    }

    val node = nodes(state)
    if (node.get.transitions(edgeIndex) == -1) {
      buildFailureLink(node.get, state, edgeIndex, char)
    }
    node.get.transitions(edgeIndex)
  }

  private def buildFailureLink(node: Node, state: Int, edgeIndex: Int, char: Char): Unit = {
    if (node.children(edgeIndex) != -1) {
      node.transitions(edgeIndex) = node.children(edgeIndex)
    } else {
      node.transitions(edgeIndex) =
        if (state == 0) 0 else findNextState(findSuffixLink(state), char)
    }
  }

  private def findSuffixLink(state: Int): Int = {
    val node = nodes(state)
    if (node.get.suffixLink == -1) {
      node.get.suffixLink =
        if (node.get.parentState == 0) 0
        else findNextState(findSuffixLink(node.get.parentState), node.get.charFromParent.get)
    }
    node.get.suffixLink
  }

  def buildAnnotation(
      chunk: ArrayBuffer[(Char, Int)],
      entity: String,
      id: String,
      sentence: Sentence): Annotation = {
    val begin = chunk.head._2 + sentence.start
    val end = chunk.last._2 + sentence.start
    val result = chunk.map(_._1).mkString("")
    val metadata = Map("entity" -> entity, "sentence" -> sentence.index.toString)

    if (id.isEmpty) {
      Annotation(CHUNK, begin, end, result, metadata)
    } else {
      Annotation(CHUNK, begin, end, result, metadata ++ Map("id" -> id))
    }

  }

  private def getAlphabetErrorMessage(char: Char): String = {
    val workshopURL = "https://github.com/JohnSnowLabs/spark-nlp/"
    val alphabetExample =
      "blob/master/examples/python/annotation/text/english/entity-ruler/EntityRuler_Alphabet.ipynb"
    val errorMessage: String =
      s"""Char $char not found in the alphabet. Your data could have unusual characters not found
         |in your document's language, which requires setting up a custom alphabet.
         |
         |Please set alphabet using setAlphabetResource parameter and make sure it has all
         |characters that can be found in your documents.
         |
         |You can check an example in Spark NLP Examples: $workshopURL$alphabetExample""".stripMargin

    errorMessage
  }

}
