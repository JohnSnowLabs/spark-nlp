/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.collections

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.TokenizerModel
import com.johnsnowlabs.nlp.annotators.btm.{TMEdgesReadWriter, TMEdgesReader, TMNodesReader, TMNodesWriter, TMVocabReadWriter, TMVocabReader, TrieNode}
import com.johnsnowlabs.storage.{Database, RocksDBConnection, StorageBatchWriter, StorageWriter}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


/**
  * Immutable Collection that used for fast substring search
  * Implementation of Aho-Corasick algorithm https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm
  */
class StorageSearchTrie(
                         vocabReader: TMVocabReader,
                         edgesReader: TMEdgesReader,
                         nodesReader: TMNodesReader
                       ) {

  /**
    * Searchs phrases in the text
    * @param text test to search in
    * @return Iterator with pairs of (begin, end)
    */
  def search(text: Seq[String]): Seq[(Int, Int)] = {
    var nodeId = 0
    val result = new ArrayBuffer[(Int, Int)]()

    def addResultIfNeed(nodeId: Int, index: Int): Unit = {
      var currentId = nodeId

      while (currentId >= 0) {
        val node = nodesReader.lookup(currentId)
        if (node.isLeaf)
          result.append((index - node.length + 1, index))

        currentId = node.lastLeaf
      }
    }

    for ((word, index) <- text.zipWithIndex) {
      val wordId = vocabReader.lookup(word).getOrElse(vocabReader.emptyValue)
      if (wordId < 0) {
        nodeId = 0
      } else {
        var found = false

        while (nodeId > 0 && !found) {
          val newId = edgesReader.lookup((nodeId, wordId)).getOrElse(edgesReader.emptyValue)
          if (newId < 0) {
            nodeId = nodesReader.lookup(nodeId).pi
          }
          else {
            nodeId = newId
            addResultIfNeed(nodeId, index)
            found = true
          }
        }

        if (!found) {
          nodeId = edgesReader.lookup((nodeId, wordId)).getOrElse(0)
          addResultIfNeed(nodeId, index)
        }
      }
    }

    result
  }
}

object StorageSearchTrie {
  def load(
            inputFileLines: Iterator[String],
            writers: Map[Database.Name, StorageWriter[_]],
            withTokenizer: Option[TokenizerModel]
          ): Unit = {

    // Have only root at the beginning
    val vocabrw = writers(Database.TMVOCAB).asInstanceOf[TMVocabReadWriter]
    var vocabSize = 0

    val edgesrw = writers(Database.TMEDGES).asInstanceOf[TMEdgesReadWriter]

    val nodesrw = writers(Database.TMNODES).asInstanceOf[TMNodesWriter]

    val parents = mutable.ArrayBuffer(0)
    val parentWord = mutable.ArrayBuffer(0)

    val isLeaf = mutable.ArrayBuffer(false)
    val length = mutable.ArrayBuffer(0)

    def vocabUpdate(w: String): Int = {
      val r = vocabrw.lookup(w).getOrElse({
        vocabrw.add(w, vocabSize)
        vocabSize
      })
      vocabSize += 1
      r
    }

    def addNode(parentNodeId: Int, wordId: Int): Int = {
      parents.append(parentNodeId)
      parentWord.append(wordId)
      length.append(length(parentNodeId) + 1)
      isLeaf.append(false)

      parents.length - 1
    }

    // Add every phrase as root from root in the tree
    for (line <- inputFileLines) {
      val phrase = withTokenizer match {
        case Some(tokenizerModel) =>
          val annotation = Seq(Annotation(line))
          tokenizerModel.annotate(annotation).map(_.result).toArray
        case _ => line.split(" ")
      }

      var nodeId = 0

      for (word <- phrase) {
        val wordId = vocabUpdate(word)
        nodeId = edgesrw.lookup((nodeId, wordId)).getOrElse({
          val r = addNode(nodeId, wordId)
          edgesrw.add((nodeId, wordId), r)
          r
        })
      }

      if (nodeId > 0)
        isLeaf(nodeId) = true
    }

    // Calculate pi function
    val piCalculated = Array.fill[Boolean](parents.size)(false)
    val pi = Array.fill[Int](parents.size)(0)

    def calcPi(v: Int): Int = {
      if (piCalculated(v))
        return pi(v)

      if (v == 0){
        piCalculated(v) = true
        pi(v) = 0
        return 0
      }

      val wordId = parentWord(v)
      var candidate = parents(v)

      while (candidate > 0) {
        candidate = calcPi(candidate)
        val answer = edgesrw.lookup((candidate, wordId)).getOrElse(0)
        if (answer > 0) {
          pi(v) = answer
          candidate = 0
        }
      }

      piCalculated(v) = true
      pi(v)
    }

    val lastLeaf = Array.fill[Int](parents.size)(-1)
    val lastLeafCalculated = Array.fill[Boolean](parents.size)(false)

    def calcLastLeaf(v: Int): Int = {
      if (lastLeafCalculated(v))
        return lastLeaf(v)

      if (v == 0) {
        lastLeafCalculated(v) = true
        lastLeaf(v) = -1
        return -1
      }

      val piNode = pi(v)
      if (isLeaf(piNode))
        lastLeaf(v) = piNode
      else
        lastLeaf(v) = calcLastLeaf(piNode)

      lastLeafCalculated(v) = true
      lastLeaf(v)
    }


    for (i <- parents.indices) {
      calcPi(i)
      calcLastLeaf(i)
    }

    pi.zip(isLeaf).zip(length).zip(lastLeaf)
      .zipWithIndex
      .foreach{case ((((a,b),c),d), i) => nodesrw.add(i, TrieNode(a,b,c,d))}

    vocabrw.close()
    edgesrw.close()
    nodesrw.close()

  }
}



