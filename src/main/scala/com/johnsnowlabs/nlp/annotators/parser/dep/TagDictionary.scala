package com.johnsnowlabs.nlp.annotators.parser

import com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition.{ClassName, ClassNum, Sentence, Word}

import scala.collection.mutable

object TagDictionary { // Here, tag == Part-of-Speech
  // Make a tag dictionary for single-tag words : So that they can be 'resolved' immediately, as well as the class list
  def classesAndTagDictionary(trainingSentences: List[Sentence]): (Vector[ClassName], Map[Word, ClassNum])  = {
    def mutationalApproach(): (mutable.Set[ClassName], mutable.Map[ Word, mutable.Map[ClassName, Int] ]) = {
      // takes 60ms on full training data !
      val classSet = mutable.Set[ClassName]()
      val fullMap  = mutable.Map[ Word, mutable.Map[ClassName, Int] ]()

      for {
        sentence <- trainingSentences
        wordData <- sentence
      } {
        classSet += wordData.pos
        fullMap.getOrElseUpdate(wordData.norm, mutable.Map[ClassName, Int]().withDefaultValue(0))(wordData.pos) += 1
      }

      (classSet, fullMap)
    }

    // First, get the set of class names, and the counts for all the words and tags
    val (classSet, fullMap) = mutationalApproach()

    // Convert the set of classes into a nice map, with indexer
    val classes = classSet.toVector.sorted  // This is alphabetical
    val classMap = classes.zipWithIndex.toMap

    val frequencyThreshold = 20
    val ambiguityThreshold = 0.97

    // Now, go through the fullMap, and work out which are worth 'resolving' immediately - and return a suitable tagdict
    val tagDictionary = mutable.Map[Word, ClassNum]().withDefaultValue(0)
    for {
      (norm, classes) <- fullMap
      if classes.values.sum >= frequencyThreshold // ignore if not enough samples
      (cl, v) <- classes
      if v >= classes.values.sum * ambiguityThreshold // Must be concentrated (in fact, cl must be unique... since >50%)
    } {
      tagDictionary(norm) = classMap(cl)
    }
    (classes, tagDictionary.toMap)
  }
}
