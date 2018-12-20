package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.Conll09Data;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;
import org.junit.Test;

import static org.junit.Assert.*;

public class Conll09ReaderTest {

    @Test
    public void shouldCreateDependencyInstanceWhenSentenceWithConll09FormatIsSent() {

        Conll09Reader conll09Reader = new Conll09Reader();
        Conll09Data[] sentence = new Conll09Data[2];
        Conll09Data wordForms = new Conll09Data("The", "the","DT", "_", 2, 0, 2);
        sentence[0] = wordForms;
        wordForms = new Conll09Data("economy", "economy","NN", "_", 4, 4,10);
        sentence[1] = wordForms;

        DependencyInstance dependencyInstance = conll09Reader.nextSentence(sentence);

        assertNotNull("dependencyInstance should not be null", dependencyInstance);
    }

    @Test
    public void shouldReturnNullWhenNoSentenceISent(){
        Conll09Reader conll09Reader = new Conll09Reader();
        Conll09Data[] sentence = new Conll09Data[1];
        Conll09Data wordForms = new Conll09Data("end", "sentence","ES", "ES", -2, 0, 0);
        sentence[0] = wordForms;
        DependencyInstance dependencyInstance = conll09Reader.nextSentence(sentence);

        assertNull("dependencyInstance should be null", dependencyInstance);
    }

}