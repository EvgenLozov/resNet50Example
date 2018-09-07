package com.company;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class Main {

    public static final String SITE_URL = "http://kiev.ukrgo.com";
    private static NativeImageLoader imageLoader = new NativeImageLoader(224  , 224);

    public static void main(String[] args) throws IOException {

        List<String> imageUrls = Arrays.asList(
                "/pictures/ukrgo_id_23795930.jpg",
                "/pictures/ukrgo_id_23795930.jpg",
                "/pictures/ukrgo_id_23795931.jpg",
                "/pictures/ukrgo_id_23795875.jpg",
                "/pictures/ukrgo_id_23795875.jpg",
                "/pictures/ukrgo_id_23795876.jpg"
        );

        ZooModel zooModel = ResNet50.builder().numClasses(100).build();

        Model net = zooModel.initPretrained(PretrainedType.IMAGENET);

        int[] inputShape = zooModel.metaData().getInputShape()[0];

        ComputationGraph graph = (ComputationGraph) zooModel.initPretrained();

        for (String imageUrl : imageUrls) {
            URL url = new URL(SITE_URL + imageUrl);
            InputStream imageInputStream = url.openStream();

            INDArray indArrayFromImage =  imageLoader.asMatrix(imageInputStream);

            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(indArrayFromImage);

            Map<String, INDArray> output = graph.feedForward(indArrayFromImage, false);


//            OutputStream os = new FileOutputStream(destinationFile);

//            byte[] b = new byte[2048];
//            int length;
//
//            while ((length = is.read(b)) != -1) {
//                os.write(b, 0, length);
//            }

            imageInputStream.close();
//            os.close();
        }
    }
}
