package com.johnsnowlabs.nlp.annotators.parser.dep.GreedyTransition

import com.johnsnowlabs.nlp.annotators.parser.dep.{Perceptron, Tagger}

/** Inspired on https://github.com/mdda/ConciseGreedyDependencyParser-in-Scala */

class DependencyMaker(tagger:Tagger) {
  val SHIFT:Move=0; val RIGHT:Move=1; val LEFT:Move=2; val INVALID:Move= -1
  def movesString(s:Set[Move]) = {
    val moveNames = Vector[ClassName]("INVALID", "SHIFT", "RIGHT", "LEFT") // NB: requires a +1
    s.toList.sorted.map( i => moveNames(i+1) ).mkString("{", ", ", "}")
  }

  private val perceptron = new Perceptron(3)

  case class ParseState(n:Int, heads:Vector[Int], lefts:Vector[List[Int]], rights:Vector[List[Int]]) { // NB: Insert at start, not at end...
    // This makes the word at 'child' point to head and adds the child to the appropriate left/right list of head
    def add(head:Int, child:Int): ParseState = {
      if(child<head) {
        ParseState(n, heads.updated(child, head), lefts.updated(head, child :: lefts(head)), rights)
      }
      else {
        ParseState(n, heads.updated(child, head), lefts, rights.updated(head, child :: rights(head)))
      }
    }
  }

  def ParseStateInit(n:Int): ParseState = {
    // heads are the dependencies for each word in the sentence, except the last one (the ROOT)
    val heads = Vector.fill(n)(0:Int) // i.e. (0, .., n-1)

    // Each possible head (including ROOT) has a (lefts) and (rights) list, initially none
    // Entries (0, ..., n-1) are words, (n) is the 'ROOT'  ('to' is INCLUSIVE)
    val lefts  = (0 to n).map( i => List[Int]() ).toVector
    val rights = (0 to n).map( i => List[Int]() ).toVector
    ParseState(n, heads, lefts, rights)
  }


  case class CurrentState(i:Int, stack:List[Int], parse:ParseState) {
    def transition(move:Move):CurrentState = move match {
      // i either increases and lengthens the stack,
      case SHIFT => CurrentState(i+1, i::stack, parse)
      // or stays the same, and shortens the stack, and manipulates left&right
      case RIGHT => CurrentState(i, stack.tail, parse.add(stack.tail.head, stack.head))   // as in Arc-Standard
      case LEFT  => CurrentState(i, stack.tail, parse.add(i, stack.head))                 // as in Arc-Eager
    }

    def getValidMoves:Set[Move] = List[Move](  // only depends on stack_depth (not parse itself)
      if(i < parse.n    )  SHIFT else INVALID, // i.e. not yet at the last word in sentence  // was n-1
      if(stack.length>=2)  RIGHT else INVALID,
      if(stack.length>=1)  LEFT  else INVALID // Original version
      //if(stack.length>=1 && stack.head != parse.n)  LEFT  else INVALID // See page 405 for second condition
    ).filterNot( _ == INVALID ).toSet

    def getGoldMoves(goldHeads:Vector[Int]): Set[Move] = {
      // See :  Goldberg and Nivre (2013) :: Training Deterministic Parsers with Non-Deterministic Oracles, TACL 2013
      //        http://www.transacl.org/wp-content/uploads/2013/10/paperno33.pdf
      //        Method implemented == "dynamic-oracle Arc-Hybrid" (bottom left of page 405, top right of page 411)
      def depsBetween(target:Int, others:List[Int]) = {
        others.exists( word => goldHeads(word)==target || goldHeads(target) == word)
      }

      val valid = getValidMoves
      //println(s"GetGold valid moves = ${moves_str(valid)}")

      if(stack.isEmpty || ( valid.contains(SHIFT) && goldHeads(i) == stack.head )) {
        //println(" gold move shortcut : {SHIFT}")
        Set(SHIFT) // First condition obvious, second rather weird
      }
      else if( goldHeads(stack.head) == i ) {
        //println(" gold move shortcut : {LEFT}")
        Set(LEFT) // This move is a must, since the gold_heads tell us to do it
      }
      else {
        // Original Python logic has been flipped over by constructing a 'val non_gold' and return 'valid - non_gold'
        //println(s" gold move logic required")

        // If the word second in the stack is its gold head, Left is incorrect
        val leftIncorrect = stack.length >= 2 && goldHeads(stack.head) == stack.tail.head

        // If there are any dependencies between i and the stack, pushing i will lose them.
        val dontPushI    = valid.contains(SHIFT) && depsBetween(i, stack) // containing SHIFT protects us against running over end of words

        // If there are any dependencies between the stackhead and the remainder of the buffer, popping the stack will lose them.
        val dontPopStack = depsBetween(stack.head, ((i+1) until parse.n).toList) // UNTIL is EXCLUSIVE of top

        val nonGold = List[Move](
          if( leftIncorrect )  LEFT  else INVALID,
          if( dontPushI )     SHIFT else INVALID,
          if( dontPopStack )  LEFT  else INVALID,
          if( dontPopStack )  RIGHT else INVALID
        ).toSet
        //println(s" gold move logic implies  : non_gold = ${moves_str(non_gold)}")

        // return the (remaining) moves, which are 'gold'
        valid -- nonGold
      }
    }

    def extractFeatures(words:Vector[Word], tags:Vector[ClassName]):Map[Feature,Score] = {
      def getStackContext[T<:String](data:Vector[T]):(T,T,T) = ( // Applies to both Word and ClassName (depth is implict from stack length)
        // NB: Always expecting 3 entries back...
        if(stack.length>0) data(stack(0)) else "".asInstanceOf[T],
        if(stack.length>1) data(stack(1)) else "".asInstanceOf[T],
        if(stack.length>2) data(stack(2)) else "".asInstanceOf[T]
      )

      def getBufferContext[T<:String](data:Vector[T]):(T,T,T) = ( // Applies to both Word and ClassName (depth is implict from stack length)
        // NB: Always expecting 3 entries back...
        if(i+0 < parse.n) data(i+0) else "".asInstanceOf[T],
        if(i+1 < parse.n) data(i+1) else "".asInstanceOf[T],
        if(i+2 < parse.n) data(i+2) else "".asInstanceOf[T]
      )

      def getParseContext[T<:String](idx:Int, deps:Vector[List[Int]], data:Vector[T]):(Int,T,T) = { // Applies to both Word and ClassName (depth is implict from stack length)
        if(idx<0) { // For the cases of empty stack
          (0, "".asInstanceOf[T], "".asInstanceOf[T])
        }
        else {
          val dependencies = deps(idx) // Find the list of dependencies at this index
          val valency = dependencies.length
          // return the tuple here :
          ( valency,
            if(valency > 0) data(dependencies(0)) else "".asInstanceOf[T],
            if(valency > 1) data(dependencies(1)) else "".asInstanceOf[T]
          )
        }
      }

      // Set up the context pieces --- the word (W) and tag (T) of:
      //   s0,1,2: Top three words on the stack
      //   n0,1,2: Next three words of the buffer (inluding this one)
      //   n0b1, n0b2: Two leftmost children of the current buffer word
      //   s0b1, s0b2: Two leftmost children of the top word of the stack
      //   s0f1, s0f2: Two rightmost children of the top word of the stack

      val n0 = i // Just for notational consistency
      val s0 = if(stack.isEmpty) -1 else stack.head

      val (ws0, ws1, ws2) = getStackContext(words)
      val (ts0, ts1, ts2) = getStackContext(tags)

      val (wn0, wn1, wn2) = getBufferContext(words)
      val (tn0, tn1, tn2) = getBufferContext(tags)

      val (vn0b, wn0b1, wn0b2) = getParseContext(n0, parse.lefts,  words)
      val (_   , tn0b1, tn0b2) = getParseContext(n0, parse.lefts,  tags)

      val (vn0f, wn0f1, wn0f2) = getParseContext(n0, parse.rights, words)
      val (_,    tn0f1, tn0f2) = getParseContext(n0, parse.rights, tags)

      val (vs0b, ws0b1, ws0b2) = getParseContext(s0, parse.lefts,  words)
      val (_,    ts0b1, ts0b2) = getParseContext(s0, parse.lefts,  tags)

      val (vs0f, ws0f1, ws0f2) = getParseContext(s0, parse.rights, words)
      val (_,    ts0f1, ts0f2) = getParseContext(s0, parse.rights, tags)

      //  String-distance :: Cap numeric features at 5? (NB: n0 always > s0, by construction)
      val dist = if(s0 >= 0) math.min(n0 - s0, 5) else 0  // WAS :: ds0n0

      val bias = Feature("bias", "")  // It's useful to have a constant feature, which acts sort of like a prior

      val wordUnigrams = for(
        word <- List(wn0, wn1, wn2, ws0, ws1, ws2, wn0b1, wn0b2, ws0b1, ws0b2, ws0f1, ws0f2)
        if(word !=0)
      ) yield Feature("w", word)

      val tagUnigrams = for(
        tag  <- List(tn0, tn1, tn2, ts0, ts1, ts2, tn0b1, tn0b2, ts0b1, ts0b2, ts0f1, ts0f2)
        if(tag !=0)
      ) yield Feature("t", tag)

      val wordTagPairs = for(
        ((word, tag), idx) <- List((wn0, tn0), (wn1, tn1), (wn2, tn2), (ws0, ts0)).zipWithIndex
        if( word!=0 || tag!=0 )
      ) yield Feature(s"wt$idx", s"w=$word t=$tag")

      val bigrams = Set(
        Feature("w s0n0", s"$ws0 $wn0"),
        Feature("t s0n0", s"$ts0 $tn0"),
        Feature("t n0n1", s"$tn0 $tn1")
      )

      val trigrams = Set(
        Feature("wtw nns", s"$wn0/$tn0 $ws0"),
        Feature("wtt nns", s"$wn0/$tn0 $ts0"),
        Feature("wtw ssn", s"$ws0/$ts0 $wn0"),
        Feature("wtt ssn", s"$ws0/$ts0 $tn0")
      )

      val quadgrams = Set(
        Feature("wtwt", s"$ws0/$ts0 $wn0/$tn0")
      )

      val tagTrigrams = for(
        ((t0,t1,t2), idx) <- List( (tn0, tn1, tn2),     (ts0, tn0, tn1),     (ts0, ts1, tn0),    (ts0, ts1, ts1),
          (ts0, ts0f1, tn0),   (ts0, ts0f1, tn0),   (ts0, tn0, tn0b1),
          (ts0, ts0b1, ts0b2), (ts0, ts0f1, ts0f2),
          (tn0, tn0b1, tn0b2)
        ).zipWithIndex
        if( t0!=0 || t1!=0 || t2!=0 )
      ) yield Feature(s"ttt-$idx", s"$t0 $t1 $t2")

      val valencyAndDistance = for(
        ((str, v), idx) <-   List( (ws0, vs0f), (ws0, vs0b), (wn0, vn0b),
          (ts0, vs0f), (ts0, vs0b), (tn0, vn0b),
          (ws0, dist), (wn0, dist), (ts0, dist), (tn0, dist),
          ("t"+tn0+ts0, dist), ("w"+wn0+ws0, dist)
        ).zipWithIndex
        if str.length>0 || v!=0
      ) yield Feature(s"val$idx", s"$str $v")

      val featureSetCombined = Set(bias) ++ bigrams ++ trigrams ++ quadgrams ++
        wordUnigrams.toSet ++ tagUnigrams.toSet ++ wordTagPairs.toSet ++
        tagTrigrams.toSet ++ valencyAndDistance.toSet

      // All weights on this set of features are ==1
      featureSetCombined.map( f => (f, 1:Score) ).toMap
    }

  }

  def train(sentences:List[Sentence], seed:Int):Float = {
    val rand = new util.Random(seed)
    rand.shuffle(sentences).map( s=>trainSentence(s) ).sum / sentences.length
  }

  def trainSentence(sentence:Sentence):Float = goodness(sentence, process(sentence, train=true))
  def parse(sentence:Sentence):List[Int] = process(sentence, train=false)

  def process(sentence:Sentence, train:Boolean):List[Int] = {
    // NB: Our structure just has a 'pure' list of sentences.  The root will point to (n)
    // Previously it was assumed that the sentence has 1 entry pre-pended, and the stack starts at {1}

    // These should be Vectors, since we're going to be accessing them at random (not sequentially)
    val words      = sentence.map( _.norm ).toVector
    val tags       = tagger.tag(sentence).toVector
    val goldHeads = sentence.map( _.dep ).toVector

    //print "train_one(n=%d, %s)" % (n, words, )
    //print " gold_heads = %s" % (gold_heads, )

    def moveThroughSentenceFrom(state: CurrentState): CurrentState = {
      val validMoves = state.getValidMoves
      if(validMoves.isEmpty) {
        state // This the answer!
      }
      else {
        //println(s"  i/n=${state.i}/${state.parse.n} stack=${state.stack}")
        val features = state.extractFeatures(words, tags)

        // This will produce scores for features that aren't valid too
        val score = perceptron.score(features, if(train) perceptron.current else perceptron.average)

        // Sort valid_moves scores into descending order, and pick the best move
        val guess = validMoves.map( m => (-score(m), m) ).toList.minBy( _._1 )._2

        if(train) {  // Update the perceptron
          //println(f"Training '${word_norm}%12s': ${classes(guessed)}%4s -> ${classes(truth(i))}%4s :: ")
          val goldMoves = state.getGoldMoves(goldHeads)
          if(goldMoves.isEmpty) { /*throw new Exception(s"No Gold Moves at ${state.i}/${state.parse.n}!")*/ } else {

            val best = goldMoves.map(m => (-score(m), m)).toList.minBy(_._1)._2
            perceptron.update(best, guess, features.keys)
          }

        }

        moveThroughSentenceFrom( state.transition(guess) )
      }
    }

    // This annotates the list of words so that parse.heads is its best guess when it finishes
    val finalState = moveThroughSentenceFrom( CurrentState(1, List(0), ParseStateInit(sentence.length)) )

    finalState.parse.heads.toList
  }

  def goodness(sentence:Sentence, fit:List[Int]):Float = {
    val gold = sentence.map( _.dep ).toVector
    val correct = fit.zip( gold ).count( pair => (pair._1 == pair._2))  / gold.length.toFloat
    correct
  }

  override def toString: String = {
    perceptron.toString()
  }

  def testGoldMoves(sentence:Sentence):Boolean = {
    val words      = sentence.map( _.norm ).toVector
    val tags       = tagger.tag(sentence).toVector
    val goldHeads = sentence.map( _.dep ).toVector

    def moveTrhoughSentenceForm(state: CurrentState): CurrentState = {
      val validMoves = state.getValidMoves
      if(validMoves.isEmpty) {
        state // This the answer!
      }
      else {
        val features = state.extractFeatures(words, tags)
        val goldMoves = state.getGoldMoves(goldHeads)
        if(goldMoves.isEmpty) {
          if(goldMoves.isEmpty) { throw new Exception(s"No Gold Moves at ${state.i}/${state.parse.n}!") }
        }
        val guess = goldMoves.toList.head
        moveTrhoughSentenceForm( state.transition(guess) )
      }
    }

    // This annotates the list of words so that parse.heads is its best guess when it finishes
    val finalState = moveTrhoughSentenceForm( CurrentState(1, List(0), ParseStateInit(sentence.length)) )

    def pct_fit_fmt_str(correct_01:Float) = {
      val correctPct = correct_01*100
      val correctStars = (0 until 100).map(i => if(i < correctPct) "x" else "-").mkString
      f"${correctPct}%6.1f%% :: $correctStars"
    }

    val correct = finalState.parse.heads.zip( goldHeads ).count( pair => pair._1 == pair._2)  / goldHeads.length
    println(s"""       index : ${goldHeads.indices.map( v => f"${v}%2d" )}""")
    println(s"""Gold   Moves : ${goldHeads.map( v => f"${v}%2d" )}""")
    println(s""" Found Moves : ${finalState.parse.heads.map( v => f"${v}%2d" )}""")
    println(f"Dependency GoldMoves correct = ${pct_fit_fmt_str(correct)}")

    correct > 0.99
  }

  def getDependencyAsArray: Iterator[String] = {
    this.toString().split(System.lineSeparator()).toIterator
  }

}

object DependencyMaker {
  def load(lines:Iterator[String], tagger:Tagger):DependencyMaker = {
    val dm = new DependencyMaker(tagger)
    dm.perceptron.load(lines)
    dm
  }

}