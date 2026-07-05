package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lattots/gonum/mat"

	"github.com/lattots/nnfs_go/pkg/neural_net"
	"github.com/lattots/nnfs_go/pkg/util"
)

const (
	learningDataFilename = "./data/mnist_train.csv"
	testingDataFilename  = "./data/mnist_test.csv"
)

func main() {
	trainingData, err := util.ReadCSVToFloat32(learningDataFilename)
	if err != nil {
		log.Fatal(err)
	}

	slicedTrainingData := trainingData[:10000]

	output, err := util.ExtractOutput(&slicedTrainingData, 0)
	if err != nil {
		log.Fatal(err)
	}

	inputLayerConfig := &neuralnet.LayerConfig{InputSize: len(slicedTrainingData[0]), NeuronCount: 32, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	firstHiddenConfig := &neuralnet.LayerConfig{NeuronCount: 32, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	secondHiddenConfig := &neuralnet.LayerConfig{NeuronCount: 32, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	thirdHiddenConfig := &neuralnet.LayerConfig{NeuronCount: 32, ActivationType: neuralnet.ReLU, InitType: neuralnet.He}
	outputLayerConfig := &neuralnet.LayerConfig{NeuronCount: 10, ActivationType: neuralnet.Softmax, InitType: neuralnet.He}

	nnConfig := neuralnet.Config{
		NumEpochs:    1000,
		LearningRate: 0.1,
		LossFunction: neuralnet.CrossEntropy,
		LayerConfigs: []*neuralnet.LayerConfig{
			inputLayerConfig,
			firstHiddenConfig,
			secondHiddenConfig,
			thirdHiddenConfig,
			outputLayerConfig,
		},
	}

	net := neuralnet.NewNet(nnConfig)
	err = net.InitRandom(10)
	if err != nil {
		log.Fatal(err)
	}

	inputMatrix, err := mat.New(slicedTrainingData)
	if err != nil {
		log.Fatal(err)
	}

	inputNormalized, err := neuralnet.NormalizeMatrix(inputMatrix)
	if err != nil {
		log.Fatal(err)
	}

	labelMatrix, err := mat.Zeros[float32](len(output), 10)
	if err != nil {
		log.Fatal(err)
	}

	for i, v := range output {
		valueInt := int(v)
		labelMatrix.Data[i*labelMatrix.N+valueInt] = 1
	}

	start := time.Now()
	err = net.Train(inputNormalized, labelMatrix)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Training took", time.Since(start))

	testingData, err := util.ReadCSVToFloat32(testingDataFilename)
	if err != nil {
		log.Fatal(err)
	}

	testingDataSliced := testingData[0:1000]

	testingOutput, err := util.ExtractOutput(&testingDataSliced, 0)
	if err != nil {
		log.Fatal(err)
	}

	successRate, err := net.Test(testingDataSliced, testingOutput)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Test complete with success rate of %.1f %%\n", successRate*100)
}
