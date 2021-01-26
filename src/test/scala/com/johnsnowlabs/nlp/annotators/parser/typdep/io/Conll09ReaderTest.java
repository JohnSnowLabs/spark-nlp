package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.ConllData;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

public class Conll09ReaderTest {

    @Test
    public void shouldCreateDependencyInstanceWhenSentenceWithConll09FormatIsSent() {

        Conll09Reader conll09Reader = new Conll09Reader();
        ConllData[] sentence = new ConllData[2];
        ConllData wordForms = new ConllData("The", "the","DT", "DT","_", 2, 0, 2);
        sentence[0] = wordForms;
        wordForms = new ConllData("economy", "economy","NN", "NN", "_", 4, 4,10);
        sentence[1] = wordForms;

        DependencyInstance dependencyInstance = conll09Reader.nextSentence(sentence);

        assertNotNull("dependencyInstance should not be null", dependencyInstance);
    }

    @Test
    public void shouldReturnNullWhenNoSentenceISent(){
        Conll09Reader conll09Reader = new Conll09Reader();
        ConllData[] sentence = new ConllData[1];
        ConllData wordForms = new ConllData("end", "sentence","ES", "ES","ES", -2, 0, 0);
        sentence[0] = wordForms;
        DependencyInstance dependencyInstance = conll09Reader.nextSentence(sentence);

        assertNull("dependencyInstance should be null", dependencyInstance);
    }

}