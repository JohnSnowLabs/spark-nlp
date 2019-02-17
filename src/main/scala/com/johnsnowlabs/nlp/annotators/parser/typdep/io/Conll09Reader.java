package com.johnsnowlabs.nlp.annotators.parser.typdep.io;

import com.johnsnowlabs.nlp.annotators.parser.typdep.ConllData;
import com.johnsnowlabs.nlp.annotators.parser.typdep.DependencyInstance;

import java.io.IOException;
import java.util.ArrayList;

public class Conll09Reader extends DependencyReader{

     /**
	     *  CoNLL 2009 format:
		    0 ID
		    1 FORM
		    2 LEMMA (not used)
		    3 PLEMMA
		    4 POS (not used)
		    5 PPOS
		    6 FEAT (not used)
		    7 PFEAT
		    8 HEAD
		    9 PHEAD
		    10 DEPREL
		    11 PDEPREL
		    12 FILLPRED
		    13 PRED
		    14... APREDn
	   	*/


    @Override
    public DependencyInstance nextInstance() throws IOException {

        ArrayList<String[]> lstLines = getFileContentAsArray();

        if (lstLines.isEmpty()) {
            return null;
        }

        int length = lstLines.size();
        String[] forms = new String[length + 1];
        String[] lemmas = new String[length + 1];
        String[] cpos = new String[length + 1];
        String[] pos = new String[length + 1];
        String[][] feats = new String[length + 1][];
        String[] deprels = new String[length + 1];
        int[] heads = new int[length + 1];

        forms[0] = "<root>";
        lemmas[0] = "<root-LEMMA>";
        pos[0] = "<root-POS>";
        cpos[0] = pos[0];
        deprels[0] = "<no-type>";
        heads[0] = -1;

        boolean hasLemma = false;

        for (int i = 1; i < length + 1; ++i) {
            String[] parts = lstLines.get(i-1);
            forms[i] = parts[1];
            if (!parts[3].equals("_")) {
                lemmas[i] = parts[3];
                hasLemma = true;
            }

            pos[i] = parts[5];
            cpos[i] = pos[i];

            if (!parts[7].equals("_")) {
                feats[i] = parts[7].split("\\|");
            }

            heads[i] = Integer.parseInt(parts[8]);
            deprels[i] = parts[10];

        }
        if (!hasLemma) lemmas = null;

        return new DependencyInstance(forms, lemmas, cpos, pos, feats, heads, deprels, null, null);
    }

    private ArrayList<String[]> getFileContentAsArray() throws IOException {

        ArrayList<String[]> lstLines = new ArrayList<>();

        String line = reader.readLine();
        while (line != null && !line.equals("") && !line.startsWith("*")) {
            lstLines.add(line.trim().split("\t"));
            line = reader.readLine();
        }
        return lstLines;
    }

}
