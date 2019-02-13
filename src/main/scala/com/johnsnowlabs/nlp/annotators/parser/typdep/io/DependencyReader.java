package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public abstract class DependencyReader {

    BufferedReader reader;

    public static DependencyReader createDependencyReader(String conllFormat) {
        if (conllFormat.equals("2009")) {
            return new Conll09Reader();
        } else {
            return new ConllUReader();
        }
    }

    public abstract DependencyInstance nextInstance() throws IOException;

    public void startReading(String file) throws IOException {
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
    }

    public void close() throws IOException { if (reader != null) reader.close(); }

}
