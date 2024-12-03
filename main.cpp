#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <cstdlib>

#include <algorithm>
#include <random>

#include <chrono>
#include <string>
#include <mach/mach.h>

#define numClasses 5

#define CARDBOARD 403
#define GlASS 501
#define METAL 410
#define PAPER 594
#define PLASTIC 482



void load (int nb, std::string type,std::vector<cv::Mat>& trainImages,std::vector<cv::Mat>& testImages,std::vector<int>& trainLabels,std::vector<int>& testLabels,std::map<std::string, int> labelMap){
///load all images of one category
    int label = labelMap[type];
    std::vector<cv::Mat> images;
    std::vector<int> labels;

    std::string datasetPath = "../ressources/dataset-resized/";

    datasetPath+=type+"/"+type;
    for (int i = 1; i<nb+1;i++){
        std::string path = datasetPath+ std::to_string(i) +".jpg";
        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        images.push_back(image);
        labels.push_back(label);
    }

    for (int i = 0; i<images.size();i++){
        if (std::rand() % 100 < 80) { // 80% for the training
            trainImages.push_back(images[i]);
            trainLabels.push_back(labels[i]);
        } else { // 20% for the test
            testImages.push_back(images[i]);
            testLabels.push_back(labels[i]);
        }
    }
}


void shuffleData(std::vector<cv::Mat>& images, std::vector<int>& labels) {
///shuffle the data to avoid having the same classes one after the other
    std::vector<size_t> indices(images.size());
    for (int i = 0; i < indices.size(); i++){
        indices[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);


    std::vector<cv::Mat> shuffledImages;
    std::vector<int> shuffledLabels;

    for (size_t i : indices) {
        shuffledImages.push_back(images[i]);
        shuffledLabels.push_back(labels[i]);
    }

    images = shuffledImages;
    labels = shuffledLabels;
}

void pretreatment(const cv::Mat& image, cv::Mat& finalFeatures) {
///prepare the image for SVM training, especially the histogram
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(64, 64));

    cv::Mat hsv;
    cv::cvtColor(resizedImage, hsv, cv::COLOR_BGR2HSV);

    // Histogramme HSV
    int hBins = 45, sBins = 45;
    int histSize[] = {hBins, sBins};
    float hRange[] = {0, 180};
    float sRange[] = {0, 256};
    const float* ranges[] = {hRange, sRange};
    int channels[] = {0, 1};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges);

    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    finalFeatures = hist.reshape(1, 1);
}


float trainAndValidate(const cv::Mat& trainingData, const std::vector<int>& labels,float C, float gamma, int kFolds = 5) {
    ///Extract validation data and labels for the current fold
    int dataSize = trainingData.rows;
    int foldSize = dataSize / kFolds;

    float totalAccuracy = 0.0;

    for (int i = 0; i < kFolds; ++i) {
        cv::Mat validationData = trainingData.rowRange(i * foldSize, (i + 1) * foldSize);
        std::vector<int> validationLabels(labels.begin() + i * foldSize, labels.begin() + (i + 1) * foldSize);

        cv::Mat trainData;
        std::vector<int> trainLabels;
        if (i > 0) {
            trainData.push_back(trainingData.rowRange(0, i * foldSize));
            trainLabels.insert(trainLabels.end(), labels.begin(), labels.begin() + i * foldSize);
        }
        if ((i + 1) * foldSize < dataSize) {
            trainData.push_back(trainingData.rowRange((i + 1) * foldSize, dataSize));
            trainLabels.insert(trainLabels.end(), labels.begin() + (i + 1) * foldSize, labels.end());
        }

        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
        svm->setType(cv::ml::SVM::C_SVC);
        svm->setKernel(cv::ml::SVM::RBF);
        svm->setC(C);
        svm->setGamma(gamma);
        cv::Mat classWeights = (cv::Mat_<float>(5, 1) << 1, 1 ,  4.77, 5.83, 1);
        svm->setClassWeights(classWeights);
        svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 1e-10));
        svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);

        float correct = 0;
        for (int j = 0; j < validationData.rows; ++j) {
            cv::Mat sample = validationData.row(j);
            int predicted = (int)svm->predict(sample);
            if (predicted == validationLabels[j]) {
                correct++;
            }
        }
        totalAccuracy += correct / validationData.rows;
    }

    return totalAccuracy / kFolds;  // Return the average precision
}

void gridSearch(const cv::Mat& trainingData, const std::vector<int>& labels) {
    /// Perform grid search to find the best hyperparameters (C and Gamma) for SVM
    float bestC = 0, bestGamma = 0;
    float bestAccuracy = 0.0;

    std::vector<float> C_values = { 0.1,1,10, 100,1000};
    std::vector<float> gamma_values = {0.001,0.01,0.1, 1,10};

    for (float C : C_values) {
        for (float gamma : gamma_values) {
            float accuracy = trainAndValidate(trainingData, labels, C, gamma);
            std::cout << "C: " << C << ", Gamma: " << gamma << ", Accuracy: " << accuracy << std::endl;

            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestC = C;
                bestGamma = gamma;
            }
        }
    }

    std::cout << "Best Parameters -> C: " << bestC << ", Gamma: " << bestGamma
              << ", Accuracy: " << bestAccuracy << std::endl;
}
/////The result is : bestC = 100 and bestGamma = 1

cv::Ptr<cv::ml::SVM> train(const cv::Mat& trainingData, const std::vector<int>& labels,cv::Ptr<cv::ml::SVM> svm) {
    ///Train the model SVM
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);

    svm->setC(100);
    svm->setGamma(1);

    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 1e-10));

    svm->train(trainingData, cv::ml::ROW_SAMPLE, labels);
    //svm->save("svm_model.xml"); ///If we want to store it
    return svm;
}


float predict(const cv::Ptr<cv::ml::SVM>& svm,const std::vector<cv::Mat>& testImages, const std::vector<int>& testLabels,float precision[numClasses],float recall[numClasses],float f1Score[numClasses]) {
    ///Predict the class of the image and write the result with the performances
    std::vector<int> truePositives(numClasses, 0);
    std::vector<int> totalGroundTruth(numClasses, 0);
    std::vector<int> totalPredictions(numClasses, 0);

    cv::Mat testData, testLabelsMat;

    for (size_t i = 0; i < testImages.size(); i++) {
        cv::Mat testSample;
        pretreatment(testImages[i], testSample);
        testSample = testSample.reshape(1, 1);
        testData.push_back(testSample);
        testLabelsMat.push_back(testLabels[i]);
    }
    testData.convertTo(testData, CV_32F);

    cv::Mat responses;
    svm->predict(testData, responses);

    responses = responses.reshape(1, 1);
    testLabelsMat = testLabelsMat.reshape(1, 1);
    responses.convertTo(responses, CV_32S);


    for (int i = 0; i < testImages.size(); i++) {
        int predictedClass = responses.at<int>(0, i);
        int trueLabel = testLabels[i];

        if (predictedClass == trueLabel) {
            truePositives[trueLabel]++;
        }
        totalGroundTruth[trueLabel]++;
        if (predictedClass >= 0) {
            totalPredictions[predictedClass]++;
        }
    }

    int correctPredictions = cv::countNonZero(responses == testLabelsMat);
    float accuracy = static_cast<float>(correctPredictions) / testLabels.size();
    //std::cout << "Global Accuracy: " << accuracy <<"\n"<< std::endl;

    for (int i = 0; i < numClasses; i++) {
        precision [i]+= totalPredictions[i] > 0 ? static_cast<float>(truePositives[i]) / totalPredictions[i] : 0;
        recall[i] += totalGroundTruth[i] > 0 ? static_cast<float>(truePositives[i]) / totalGroundTruth[i] : 0;
        //f1Score [i]+= (precision[i] + recall[i]> 0) ? (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) : 0;

        /*std::cout << "Class " << i << " - Precision: " << precision[i]
                  << ", Recall: " << recall[i] << ", F1-Score: " << f1Score[i] << std::endl;*/
    }

    return accuracy;
}

size_t getMemoryUsage() {
    /// Get the memory in byte
    mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
}




void test (float& accurate,float precision[numClasses], float recall[numClasses], float f1Score[numClasses],long& trainDuration,size_t ramUsed[2]){

    size_t memoryBefore = getMemoryUsage();

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> trainImages,testImages;
    std::vector<int> trainLabels,testLabels;

    // Label dictionary
    std::map<std::string, int> labelMap = {
            {"plastic", 0},
            {"paper", 1},
            {"glass", 2},
            {"metal", 3},
            {"cardboard", 4}
    };

    std::srand(std::time(nullptr));

    load(PLASTIC,"plastic",trainImages,testImages,trainLabels,testLabels,labelMap);
    load(CARDBOARD,"cardboard",trainImages,testImages,trainLabels,testLabels,labelMap);
    load(GlASS,"glass",trainImages,testImages,trainLabels,testLabels,labelMap);
    load(METAL,"metal",trainImages,testImages,trainLabels,testLabels,labelMap);
    load(PAPER,"paper",trainImages,testImages,trainLabels,testLabels,labelMap);

    shuffleData(trainImages, trainLabels);
    shuffleData(testImages, testLabels);


    cv::Mat trainingData;

    for (int i = 0;i<trainImages.size();i++) {
        cv::Mat features;
        pretreatment(trainImages[i],features);
        trainingData.push_back(features);
    }
    trainingData.convertTo(trainingData, CV_32F);

    //gridSearch(trainingData, trainLabels);

   // cv::Mat classWeights = (cv::Mat_<float>(5, 1) << 4.96, 4.02 , 4.77, 5.83, 5.93);
    cv::Mat classWeights = (cv::Mat_<float>(5, 1) << 1, 1 ,  4.77, 5.83, 1);
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setClassWeights(classWeights);


    svm = train(trainingData,trainLabels,svm );
    size_t memoryAfter = getMemoryUsage();
    ramUsed[0] += memoryAfter - memoryBefore;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    trainDuration+=duration;

    memoryBefore = getMemoryUsage();
    accurate += predict(svm, testImages, testLabels,precision,recall,f1Score);
    memoryAfter = getMemoryUsage();
    ramUsed[1] += memoryAfter - memoryBefore;
}

int main() {
    float precision[numClasses], recall[numClasses],  f1Score[numClasses];
    for (int i = 0;i<numClasses;i++){
        precision[i] = 0;
        recall[i] = 0;
    }
    long trainDuration = 0;
    size_t ramUsed[2] ={0,0};

    const int lenght = 5;
    auto start = std::chrono::high_resolution_clock::now();
    float average = 0;

    std::cout << "\nWe execute the code "<<lenght<< " times for a more accurate average of model and training performance\n"<< std::endl;
    for (int i = 0;i<lenght;i++){
        test(average,precision,recall,f1Score,trainDuration,ramUsed);
   }
    average = average/lenght;
    ramUsed[0] = ramUsed[0]/lenght;
    ramUsed[1] = ramUsed[1]/lenght;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution time for all : " << duration.count()*0.001 << " s"  << " for each : "<<(duration.count()/lenght)*0.001<< " s" << std::endl;
    std::cout << "Train duration time for all : " << trainDuration*0.001 << " s"  << " for each : "<<(trainDuration/lenght)*0.001<< " s" << std::endl;
    std::cout << "Test duration time for all : " << (duration.count()-trainDuration)*0.001 << " s"  << " for each : "<<(duration.count()/lenght -trainDuration/lenght)*0.001<< " s" << std::endl;
    std::cout << "\nMemory used for train : " << ramUsed[0]/ (1024*1024) << " MB"<< std::endl;
    std::cout << "Memory used for test : " << ramUsed[1]/ (1024*1024) << " MB"<< std::endl;

    std::cout << "\nAverage all class include : " << average  <<"\nAverage for each class : "<<std::endl;

    for (int i = 0;i<numClasses;i++) {
        precision[i] = precision[i]/lenght;
        recall[i] = recall[i]/lenght;
        f1Score [i]= (precision[i] + recall[i]> 0) ? (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) : 0;
        std::cout << "Class " << i << " - Precision: " << precision[i] << ", Recall: " << recall[i] << ", F1-Score: "
                  << f1Score[i] << std::endl;
    }
    return 0;
}