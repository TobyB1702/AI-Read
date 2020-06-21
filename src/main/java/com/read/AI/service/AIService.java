package com.read.AI.service;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jetbrains.annotations.NotNull;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.ExpandDims;
import org.nd4j.linalg.api.ops.impl.shape.Reshape;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;


//multilayered (MLP) applied to digit classification
//Uses Two input layers and one hidden layer
public class AIService {

    private static final int numRows = 28;
    private static final int numColumns = 28;
    private static final int outputNum = 10;// number of output classes
    private static final int batchSize = 64;// batch size for each epoch
    private static final int rngSeed = 123;// random number seed for reproducibility
    private static final int numEpochs = 15;// number of epochs to perform
    private static final double rate = 0.0015;// learning rate
    private static final String FILE_PATH = "AI-Read-Model.zip";
    private static final boolean saveUpdater = true;
    private static final File locationToSave = new File(FILE_PATH);

    private static Logger log = LoggerFactory.getLogger(AIService.class);

    public void DoesModelFileExist(String filePath) throws IOException {
        if (locationToSave.exists()) {
            log.info("Saved Model Found!");

        } else {
            log.error("File not found!");
            log.error("Creating Model Now");
            setupModelAndData();
        }
    }

    public int[] FiveImageTest() throws IOException {
        MultiLayerNetwork loadedModel = MultiLayerNetwork.load(locationToSave, saveUpdater);
        BufferedImage myImage = ImageIO.read(new FileInputStream("fiveTestImage.png"));

        NativeImageLoader loader = new NativeImageLoader(myImage.getHeight(), myImage.getWidth(), 1);
        ImagePreProcessingScaler imageScaler = new ImagePreProcessingScaler();

        INDArray img = loader.asMatrix(myImage);
        INDArray reshaped = img.reshape(1,784);

        imageScaler.transform(reshaped);

        INDArray output = loadedModel.output(reshaped,true);
        log.info(output.toString());
        final int[] lastLayerOfNetwork = loadedModel.predict(reshaped);

        return lastLayerOfNetwork;
    }

    public static void setupModelAndData() throws IOException {
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,true,rngSeed);

        MultiLayerNetwork model = buildModel(numRows, numColumns, outputNum, rngSeed, rate);

        log.info("Train model....");
        model.fit(mnistTrain, numEpochs);
        log.info("Evaluate model....");
        Evaluation eval = model.evaluate(mnistTest);
        log.info(eval.stats());
        log.info("****************modelFinished********************");
        log.info("Testing model on example image....");
        log.info("SAVE TRAINED MODEL");

        //Save the model
        File locationToSave = new File(FILE_PATH);
        model.save(locationToSave, saveUpdater);
    }

    private static MultiLayerConfiguration setConfigurationForModel(int numRows, int numColumns, int outputNum, int rngSeed, double rate) {
        log.info("Setting up Configuration for Model....");
        return new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nadam())
                .l2(rate * 0.005) // Regularize Learning model
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numRows * numColumns)
                        .nOut(500)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(500)
                        .nOut(100)
                        .build())
                .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nOut(outputNum)
                        .build())
                .build();
    }

    @NotNull
    private static MultiLayerNetwork buildModel(int numRows, int numColumns, int outputNum, int rngSeed, double rate) throws IOException {
        log.info("Build model....");
        MultiLayerConfiguration configuration = setConfigurationForModel(numRows, numColumns, outputNum, rngSeed, rate);
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.setListeners(new ScoreIterationListener(5));//print the score with every iteration

        return model;
    }



}
