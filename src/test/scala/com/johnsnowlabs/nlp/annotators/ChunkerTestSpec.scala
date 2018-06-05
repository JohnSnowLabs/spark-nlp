package com.johnsnowlabs.nlp.annotators

import org.scalatest.FlatSpec

class ChunkerTestSpec extends FlatSpec with ChunkerBehaviors{

  "a trained POS tag" should behave like testPOSForChunking(
  Array(
  "first sentence example",
  "second something going"
  )
  )

  "a chunk" should behave like testPOSForChunkingWithPipeline(
  Array(
  "first sentence example",
  "second something going"
  )
  )

  "a chunk with user input tags" should behave like testUserInputPOSTags(
    Array("the little yellow dog barked at the cat"),
    Array(Array("DT", "JJ", "JJ", "NN", "VBD", "IN", "DT", "NN"))
  )

//  sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), [1]
//  ... ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

}
