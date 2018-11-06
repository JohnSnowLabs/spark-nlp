package com.johnsnowlabs.nlp.annotators.parser.typdep;

import com.johnsnowlabs.nlp.annotators.parser.typdep.io.Conll09Writer;
import com.johnsnowlabs.nlp.annotators.parser.typdep.util.DependencyLabel;

import java.io.*;
import java.util.ArrayList;

public class TypedDependencyParser implements Serializable {

    private static final long serialVersionUID = 1L;

    private Options options;
    private DependencyPipe dependencyPipe;
    private Parameters parameters;

    DependencyPipe getDependencyPipe() {
        return dependencyPipe;
    }

    Parameters getParameters() {
        return parameters;
    }

    public Options getOptions() {
        return options;
    }

    void setDependencyPipe(DependencyPipe dependencyPipe) {
        this.dependencyPipe = dependencyPipe;
    }

    void setParameters(Parameters parameters) {
        this.parameters = parameters;
    }

    public void setOptions(Options options) {
        this.options = options;
    }

    void train(DependencyInstance[] dependencyInstances)
    {
        long start;
        long end;

        if ((options.rankFirstOrderTensor > 0 || options.rankSecondOrderTensor > 0) && options.gammaLabel < 1 && options.initTensorWithPretrain) {

            Options optionsBackup = Options.newInstance(options);
            options.rankFirstOrderTensor = 0;
            options.rankSecondOrderTensor = 0;
            options.gammaLabel = 1.0f;
            optionsBackup.numberOfTrainingIterations = options.numberOfPreTrainingIterations;
            parameters.setRankFirstOrderTensor(options.rankFirstOrderTensor);
            parameters.setRankSecondOrderTensor(options.rankSecondOrderTensor);
            parameters.setGammaLabel(options.gammaLabel);

            System.out.printf("Pre-training:%n");

            start = System.currentTimeMillis();

            System.out.println("Running MIRA ... ");
            trainIterations(dependencyInstances);
            System.out.println();

            options = optionsBackup;
            parameters.setRankFirstOrderTensor(options.rankFirstOrderTensor);
            parameters.setRankSecondOrderTensor(options.rankSecondOrderTensor);
            parameters.setGammaLabel(options.gammaLabel);

            System.out.println("Init tensor ... ");
            int n = parameters.getNumberWordFeatures();
            int d = parameters.getDL();
            LowRankTensor tensor = new LowRankTensor(new int[] {n, n, d}, options.rankFirstOrderTensor);
            LowRankTensor tensor2 = new LowRankTensor(new int[] {n, n, n, d, d}, options.rankSecondOrderTensor);
            dependencyPipe.getSynFactory().fillParameters(tensor, tensor2, parameters);

            ArrayList<float[][]> param = new ArrayList<>();
            param.add(parameters.getU());
            param.add(parameters.getV());
            param.add(parameters.getWL());
            tensor.decompose(param);
            ArrayList<float[][]> param2 = new ArrayList<>();
            param2.add(parameters.getU2());
            param2.add(parameters.getV2());
            param2.add(parameters.getW2());
            param2.add(parameters.getX2L());
            param2.add(parameters.getY2L());
            tensor2.decompose(param2);
            parameters.assignTotal();
            parameters.printStat();

            System.out.println();
            end = System.currentTimeMillis();
            System.out.println();
            System.out.printf("Pre-training took %d ms.%n", end-start);
            System.out.println();

        } else {
            parameters.randomlyInit();
        }

        System.out.printf(" Training:%n");

        start = System.currentTimeMillis();

        System.out.println("Running MIRA ... ");
        trainIterations(dependencyInstances);
        System.out.println();

        end = System.currentTimeMillis();

        System.out.printf("Training took %d ms.%n", end-start);
        System.out.println();
    }

    private void trainIterations(DependencyInstance[] dependencyInstances)
    {
        int printPeriod = 10000 < dependencyInstances.length ? dependencyInstances.length/10 : 1000;

        System.out.println("***************************************************** Number of Training Iterations: "+options.numberOfTrainingIterations);

        for (int iIter = 0; iIter < options.numberOfTrainingIterations; ++iIter) {

            long start;
            double loss = 0;
            int totalNUmberCorrectMatches = 0;
            int tot = 0;
            start = System.currentTimeMillis();

            for (int i = 0; i < dependencyInstances.length; ++i) {

                if ((i + 1) % printPeriod == 0) {
                    System.out.printf("  %d (time=%ds)", (i+1),
                            (System.currentTimeMillis()-start)/1000);
                }

                DependencyInstance dependencyInstance = dependencyInstances[i];
                LocalFeatureData localFeatureData = new LocalFeatureData(dependencyInstance, this);
                int dependencyInstanceLength = dependencyInstance.getLength();
                int[] predictedHeads = dependencyInstance.getHeads();
                int[] predictedLabels = new int [dependencyInstanceLength];

                localFeatureData.predictLabels(predictedHeads, predictedLabels, true);
                int numberCorrectMatches = getNumberCorrectMatches(dependencyInstance.getHeads(),
                        dependencyInstance.getDependencyLabelIds(),
                        predictedHeads, predictedLabels);
                if (numberCorrectMatches != dependencyInstanceLength-1) {
                    loss += parameters.updateLabel(dependencyInstance, predictedHeads, predictedLabels,
                            localFeatureData, iIter * dependencyInstances.length + i + 1);
                }
                totalNUmberCorrectMatches += numberCorrectMatches;
                tot += dependencyInstanceLength-1;
            }

            tot = tot == 0 ? 1 : tot;

            System.out.printf("%n  Iter %d\tloss=%.4f\ttotalNUmberCorrectMatches" +
                            "=%.4f\t[%ds]%n", iIter+1,
                    loss, totalNUmberCorrectMatches
                            /(tot +0.0),
                    (System.currentTimeMillis() - start)/1000);
            System.out.println();

            parameters.printStat();
        }

    }

    private int getNumberCorrectMatches(int[] actualHeads, int[] actualLabels, int[] predictedHeads, int[] predictedLabels)
    {
        int nCorrect = 0;
        for (int i = 1, N = actualHeads.length; i < N; ++i) {
            if (actualHeads[i] == predictedHeads[i] && actualLabels[i] == predictedLabels[i])
                ++nCorrect;
        }
        return nCorrect;
    }

    DependencyLabel[] predictDependency(Conll09Data[][] document){

        Conll09Writer conll09Writer = new Conll09Writer(options, dependencyPipe);

        DependencyLabel[] dependencyLabels = new DependencyLabel[document[0].length];

        for (Conll09Data[] sentence : document) {
            DependencyInstance dependencyInstance = dependencyPipe.nextSentence(sentence);
            if (dependencyInstance == null) {
                break;
            }
            LocalFeatureData localFeatureData = new LocalFeatureData(dependencyInstance, this);
            int numberOfTokensInSentence = dependencyInstance.getLength();
            int[] predictedHeads = dependencyInstance.getHeads();
            int[] predictedLabels = new int [numberOfTokensInSentence];
            localFeatureData.predictLabels(predictedHeads, predictedLabels, false);

            dependencyLabels = conll09Writer.getDependencyLabels(dependencyInstance, predictedHeads, predictedLabels);
        }
        return dependencyLabels;
    }


}
