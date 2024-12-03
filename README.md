# Wast-recognition-and-sorting

## Project Results and Overview

### Key objectives
The project aims to design and implement an automated waste classification system using Support Vector Machines (SVM) and OpenCV. The goal is to accurately classify five types of recyclable waste : plastic, paper, glass, metal, and cardboard, based on their visual features. The motivation stems from environmental concerns and the desire to automate recycling processes, reducing manual labor and improving sorting efficiency.
The aim was also to get a feel for C++, because I'm interested in creating video games and I was still stuck on C. <br/> 
My program is made for testing and find the best solutions. To do this, I run my program 5 times and average the results I receive so that the data I receive is smoothed and I can see instantly when a new solution is better or not.

### Results
The results are the average for one train and test not for 5.<br/>
The SVM model achieves an accuracy of 73% on the test set.
The model training process is completing in 5 seconds using a dataset of 1600 images.
The testing phase processes the remaining 400 images in only 0.3 seconds.
During training, the program uses approximately 550 MB of RAM.
During testing, memory usage is 100 MB.




## Source Code
```bash
/Project
├── CMakeLists.txt        # CMake configuration file for building the project
├── README.md             # Project documentation (overview, setup, usage, etc.)
├── main.cpp              # Main source file with the program's entry point
├── /ressources      # Directory containing resized datasets, model files and graphics
│   ├── svm_model.xml     # Pretrained SVM model file for classification
│   ├── performances # Performance graphs of this project 
│       ├── RAM.png
│       ├── Execute_time.png
│       └── Global_performances.png
│   └── /dataset          # Dataset organized by material type
│       ├── /cardboard-resized    # Images of cardboard for training/testing
│       │   ├── cardboard1.jpg
│       │   └── cardboard403.jpg
│       ├── /glass        # Images of glass for training/testing
│       │   ├── glass1.jpg
│       │   └── glass501.jpg
│       ├── /metal        # Images of metal for training/testing
│       │   ├── metal1.jpg
│       │   └── metal410.jpg
│       ├── /paper        # Images of paper for training/testing
│       │   ├── paper1.jpg
│       │   └── paper594.jpg
│       └── /plastic      # Images of plastic for training/testing
│           ├── plastic1.jpg
│           └── plastic482.jpg
```

### Performance Metrics
For the first graphic we will see the RAM used by the program for the training and testing, we can see that the testing use much less memory than the training.<br/>
![Screenshot 2024-12-03 at 21.11.44.png](/ressources/performances/RAM.png)
<br/><br/>This one is the graph showing the time taken by the program for training and testing. The time for testing is more than 10 times shorter than the time for training.  <br/>
![Screenshot 2024-12-03 at 21.11.44.png](/ressources/performances/Execute_time.png)
<br/><br/> This graph shows the precision of the program, and we can see that the glass and especially the metal class performs much less well than the others, which I haven't been able to increase.<br/>
![Screenshot 2024-12-03 at 21.11.44.png](/ressources/performances/Global_performances.png)

## Installation and Usage

For this project we use OpenCV in C++, I'm on macOS, so I hope it will also work on windows, here is the installation for both OS :
All the operations are on the terminal.

### MacOS
1. We need to install Homebrew : 
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. After we need to install openCV :

```bash
brew install opencv
```
3. Finally, we check that the installation has gone well :

```bash
pkg-config --modversion opencv4
```
### Windows
1. Clone the vcpkg repository from GitHub
```bash
git clone https://github.com/microsoft/vcpkg.git
```
2. Go to the vcpkg directory
```bash
cd vcpkg
```
3. Install vcpkg
```bash
.\bootstrap-vcpkg.bat
```

4. Install OpenCV with vcpkg
```bash
.\vcpkg install opencv

```
5. To automatically include OpenCV in our project, we use this command
```bash
.\vcpkg integrate install
```

### Usage
We simply need to execute the project, there’s nothing else to do.


## References and Documentation
For the dataset I downloaded this one : https://www.kaggle.com/datasets/feyzazkefe/trashnet?resource=download<br/>
For the documentation on OpenCV in general :  https://docs.opencv.org/4.x/index.html<br/>
For the documentation on SVM models : https://docs.opencv.org/4.x/d1/d73/tutorial_introduction_to_svm.html

## Issues and Contributions
The current SVM model achieves only about 73% accuracy, which is a significant limitation. This is largely due to the small dataset, which consists of only 2000 images spread across 5 classes.
While the small dataset is a disadvantage for professional use, it offers an interesting educational opportunity. The small size allows me for very fast training (around 5 seconds), which is ideal for testing and learning how to improve training techniques. However, for practical or professional applications, the dataset size is insufficient.
Some classes, particularly metal, consistently perform poorly in classification. Despite my efforts to improve its accuracy, the model struggles with this class, likely due to insufficient or less representative images in the dataset.
Only five classes are currently supported (cardboard, glass, metal, paper, and plastic). Expanding the dataset to include more materials is necessary for broader applicability.
<br/><br/>
For the contribution, we need to ensure that provide more images or diverse datasets for the existing classes, or add new materials. Or also, experiment with different machine learning algorithms or optimize the current SVM approach to improve classification accuracy, particularly for the underperformed metal class.

## Future Work
To improve the project and make it more applicable in practical scenarios, several enhancements are planned:

The goal is to improve the model’s performance by at least 20%, reaching 90% accuracy or higher. This will likely require a combination of techniques, including expanding the dataset or improving feature extraction.
An important step is to distinguish between different types of plastics to determine which are recyclable and which are not. This is a highly complex and relevant challenge in waste management today. Successfully implementing this feature would significantly increase the project's value and impact.
Another key area for future development is enabling real-time waste classification. This would involve optimizing the program to process video streams from a camera efficiently. Such a feature could pave the way for integration into industrial settings, like sorting facilities in recycling plants.
These enhancements aim to bridge the gap between the current educational focus of the project and its potential for real-world applications.