package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.feature.SyntacticFeatureFactory;
import com.johnsnowlabs.nlp.annotators.parser.typdep.io.DependencyReader;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Dictionary;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import static com.johnsnowlabs.nlp.annotators.parser.typdep.feature.FeatureTemplate.Arc.NUM_ARC_FEAT_BITS;
import static com.johnsnowlabs.nlp.annotators.parser.typdep.feature.FeatureTemplate.Word.NUM_WORD_FEAT_BITS;
import static com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet.DictionaryTypes.DEP_LABEL;
import static com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet.DictionaryTypes.POS;
import static com.johnsnowlabs.nlp.annotators.parser.typdep.util.DictionarySet.DictionaryTypes.WORD;

public class DependencyPipe implements Serializable {

    private static final long serialVersionUID = 1L;

    private Options options;
    private DictionarySet dictionariesSet;
    private SyntacticFeatureFactory synFactory;

    public Options getOptions() {
        return options;
    }

    public void setOptions(Options options) {
        this.options = options;
    }

    public DictionarySet getDictionariesSet() {
        return dictionariesSet;
    }

    SyntacticFeatureFactory getSynFactory() {
        return synFactory;
    }

    public void setDictionariesSet(DictionarySet dictionariesSet) {
        this.dictionariesSet = dictionariesSet;
    }

    private String[] types;					// array that maps label index to label string

    public String[] getTypes() {
        return types;
    }

    // language specific info
    private HashSet<String> conjWord;
    private HashMap<String, String> coarseMap;

    // headPOS x modPOS x Label
    private boolean[][][] pruneLabel;

    boolean[][][] getPruneLabel() {
        return pruneLabel;
    }

    private int numCPOS;

    DependencyPipe(Options options)
    {
        dictionariesSet = new DictionarySet();
        synFactory = new SyntacticFeatureFactory();

        this.options = options;

        loadLanguageInfo();
    }

    DependencyPipe(Options options, DictionarySet dictionariesSet, SyntacticFeatureFactory synFactory){
        options = options;
        dictionariesSet = dictionariesSet;
        synFactory = synFactory;
    }

    /***
     * load language specific information
     * conjWord: word considered as a conjunction
     * coarseMap: fine-to-coarse map
     */
    private void loadLanguageInfo() {
        // load coarse map
        coarseMap = new HashMap<>();
        try {
            try (BufferedReader br = new BufferedReader(new FileReader(options.unimapFile))) {
                String str;
                while ((str = br.readLine()) != null) {
                    String[] data = str.split("\\s+");
                    coarseMap.put(data[0], data[1]);
                }
            }

            coarseMap.put("<root-POS>", "ROOT");
        } catch (Exception e) {
            System.out.println("Warning: couldn't find coarse POS map for this language");
        }

        // fill conj word
        conjWord = new HashSet<>();
        conjWord.add("and");
        conjWord.add("or");

    }

    /***F
     * Build dictionariesSet that maps word strings, POS strings, etc into
     * corresponding integer IDs. This method is called before creating
     * the feature alphabets and before training a dependency model.
     *
     * @param file file path of the training data
     */
    private void createDictionaries(String file) throws IOException
    {

        long start = System.currentTimeMillis();
        System.out.println("Creating dictionariesSet ... ");

        dictionariesSet.setCounters();

        DependencyReader reader = DependencyReader.createDependencyReader(options);

        reader.startReading(file);
        DependencyInstance dependencyInstance = reader.nextInstance(coarseMap);

        while (dependencyInstance != null) {
            //This loop sets values in dictionariesSet for later use
            dependencyInstance.setInstIds(dictionariesSet, coarseMap, conjWord);

            dependencyInstance = reader.nextInstance(coarseMap);
        }
        reader.close();

        dictionariesSet.closeCounters();

        synFactory.setTokenStart(dictionariesSet.lookupIndex(POS, "#TOKEN_START#"));
        synFactory.setTokenEnd(dictionariesSet.lookupIndex(POS, "#TOKEN_END#"));
        synFactory.setTokenMid(dictionariesSet.lookupIndex(POS, "#TOKEN_MID#"));

        dictionariesSet.stopGrowth(DEP_LABEL);
        dictionariesSet.stopGrowth(POS);
        dictionariesSet.stopGrowth(WORD);

        synFactory.setWordNumBits(Utils.log2((long) dictionariesSet.getDictionarySize(WORD) + 1));
        synFactory.setTagNumBits(Utils.log2((long) dictionariesSet.getDictionarySize(POS) + 1));
        synFactory.setDepNumBits(Utils.log2((long) dictionariesSet.getDictionarySize(DEP_LABEL) + 1));
        synFactory.setFlagBits(2*synFactory.getDepNumBits() + 4);

        types = new String[dictionariesSet.getDictionarySize(DEP_LABEL)];
        Dictionary labelDict = dictionariesSet.getDictionary(DEP_LABEL);
        Object[] keys = labelDict.toArray();
        for (Object key : keys) {
            int id = labelDict.lookupIndex(key);
            types[id - 1] = (String) key;
        }

        System.out.printf("%d %d%n", NUM_WORD_FEAT_BITS, NUM_ARC_FEAT_BITS);
        System.out.printf("Lexical items: %d (%d bits)%n",
                dictionariesSet.getDictionarySize(WORD), synFactory.getWordNumBits());
        System.out.printf("Tag/label items: %d (%d bits)  %d (%d bits)%n",
                dictionariesSet.getDictionarySize(POS), synFactory.getTagNumBits(),
                dictionariesSet.getDictionarySize(DEP_LABEL), synFactory.getDepNumBits());
        System.out.printf("Flag Bits: %d%n", synFactory.getFlagBits());
        System.out.printf("Creation took [%d ms]%n", System.currentTimeMillis() - start);
    }

    /***
     * Create feature alphabets, which maps 64-bit feature code into
     * its integer index (starting from index 0). This method is called
     * before training a dependency model.
     *
     * @param file  file path of the training data
     */
    public void createAlphabets(String file) throws IOException
    {

        createDictionaries(file);

        long start = System.currentTimeMillis();
        System.out.print("Creating Alphabet ... ");

        HashSet<String> posTagSet = new HashSet<>();
        HashSet<String> cposTagSet = new HashSet<>();
        DependencyReader reader = DependencyReader.createDependencyReader(options);
        reader.startReading(file);

        DependencyInstance dependencyInstance = reader.nextInstance(coarseMap);

        while(dependencyInstance != null) {

            for (int i = 0; i < dependencyInstance.getLength(); ++i) {
                if (dependencyInstance.getPostags() != null) posTagSet.add(dependencyInstance.getPostags()[i]);
                if (dependencyInstance.getCpostags() != null) cposTagSet.add(dependencyInstance.getCpostags()[i]);
            }

            dependencyInstance.setInstIds(dictionariesSet, coarseMap, conjWord);

            synFactory.initFeatureAlphabets(dependencyInstance);

            dependencyInstance = reader.nextInstance(coarseMap);
        }

        System.out.printf("[%d ms]%n", System.currentTimeMillis() - start);

        closeAlphabets();
        reader.close();

        synFactory.checkCollisions();
        System.out.printf("Num of CONLL fine POS tags: %d%n", posTagSet.size());
        System.out.printf("Num of CONLL coarse POS tags: %d%n", cposTagSet.size());
        System.out.printf("Num of labels: %d%n", types.length);
        System.out.printf("Num of Syntactic Features: %d %d%n",
                synFactory.getNumberWordFeatures(), synFactory.getNumberLabeledArcFeatures());
        numCPOS = cposTagSet.size();
    }

    /***
     * Close alphabets so the feature set wouldn't grow.
     */
    public void closeAlphabets()
    {
        synFactory.closeAlphabets();
    }

    public DependencyInstance[] createInstances(String file) throws IOException
    {

        long start = System.currentTimeMillis();
        System.out.print("Creating instances ... ");

        DependencyReader reader = DependencyReader.createDependencyReader(options);
        reader.startReading(file);

        LinkedList<DependencyInstance> lt = new LinkedList<>();
        DependencyInstance dependencyInstance = reader.nextInstance(coarseMap);

        while(dependencyInstance != null) {

            dependencyInstance.setInstIds(dictionariesSet, coarseMap, conjWord);

            lt.add(new DependencyInstance(dependencyInstance));

            dependencyInstance = reader.nextInstance(coarseMap);
        }

        reader.close();
        closeAlphabets();

        DependencyInstance[] insts = new DependencyInstance[lt.size()];
        int N = 0;
        for (DependencyInstance p : lt) {
            insts[N++] = p;
        }

        return insts;
    }

    public DependencyInstance createInstance(DependencyReader reader) throws IOException
    {

        DependencyInstance inst = reader.nextInstance(coarseMap);
        if (inst == null) return null;

        inst.setInstIds(dictionariesSet, coarseMap, conjWord);

        return inst;
    }

    public void pruneLabel(DependencyInstance[] dependencyInstances)
    {
        int numPOS = dictionariesSet.getDictionarySize(POS) + 1;
        int numLab = dictionariesSet.getDictionarySize(DEP_LABEL) + 1;
        this.pruneLabel = new boolean [numPOS][numPOS][numLab];
        int num = 0;

        for (DependencyInstance dependencyInstance : dependencyInstances) {
            int n = dependencyInstance.getLength();
            for (int mod = 1; mod < n; ++mod) {
                int head = dependencyInstance.getHeads()[mod];
                int lab = dependencyInstance.getDependencyLabelIds()[mod];
                if (!this.pruneLabel[dependencyInstance.getCpostagids()[head]][dependencyInstance.getCpostagids()[mod]][lab]) {
                    this.pruneLabel[dependencyInstance.getCpostagids()[head]][dependencyInstance.getCpostagids()[mod]][lab] = true;
                    num++;
                }
            }
        }

        System.out.println("Prune label: " + num + "/" + numCPOS*numCPOS*numLab);
    }

}
