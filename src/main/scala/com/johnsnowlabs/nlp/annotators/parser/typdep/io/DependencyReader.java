package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.ConllData;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

public abstract class DependencyReader {

    private static final int END_OF_SENTENCE = -2;
    BufferedReader reader;

    public static DependencyReader createDependencyReader(String conllFormat) {
        if (conllFormat.equals("2009")) {
            return new Conll09Reader();
        } else {
            return new ConllUReader();
        }
    }

    public abstract DependencyInstance nextInstance() throws IOException;

    public DependencyInstance nextSentence(ConllData[] sentence) {

        if (sentence[0].getHead() == END_OF_SENTENCE) {
            return null;
        }

        int length = sentence.length;
        String[] forms = new String[length + 1];
        String[] lemmas = new String[length + 1];
        String[] cpos = new String[length + 1];
        String[] pos = new String[length + 1];
        String[][] feats = new String[length + 1][];
        String[] deprels = new String[length + 1];
        int[] heads = new int[length + 1];
        int[] begins = new int[length + 1];
        int[] ends = new int[length + 1];

        forms[0] = "<root>";
        lemmas[0] = "<root-LEMMA>";
        pos[0] = "<root-POS>";
        cpos[0] = pos[0];
        deprels[0] = "<no-type>";
        heads[0] = -1;
        begins[0] = -1;
        ends[0] = -1;

        boolean hasLemma = false;

        for (int i = 1; i < length + 1; ++i) {
            ConllData conllValues = sentence[i-1];
            begins[i] = conllValues.getBegin();
            ends[i] = conllValues.getEnd();
            forms[i] = conllValues.getForm();

            if (!conllValues.getLemma().equals("_")) {
                lemmas[i] = conllValues.getLemma();
                hasLemma = true;
            }

            pos[i] = conllValues.getPos();
            cpos[i] = pos[i];

            //TODO: Add feats

            heads[i] = conllValues.getHead();
            deprels[i] = conllValues.getDeprel();

        }
        if (!hasLemma){
            lemmas = null;
        }

        return new DependencyInstance(forms, lemmas, cpos, pos, feats, heads, deprels, begins, ends);
    }

    public void startReading(String file) throws IOException {
        reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8));
    }

    public void close() throws IOException { if (reader != null) reader.close(); }

}
