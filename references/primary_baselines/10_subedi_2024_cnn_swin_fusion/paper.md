# Classification of Endoscopy and Video Capsule Images using CNN-Transformer Model

Aliza Subedi 1 , Smriti Regmi 1 , Nisha Regmi 2 , Bhumi Bhusal 2 , Ulas Bagci 2 , and Debesh Jha 2

$^{1}$ Pashchimanchal Campus, Nepal $^{2}$ Machine & Hybrid Intelligence Lab, Department of Radiology, Northwestern University, Chicago, USA

Abstract. Gastrointestinal cancer is the leading cause of cancer-related incidence and death. Therefore, it is important to develop a novel computeraided diagnosis system for early detection and enhanced treatment. Traditional approaches rely on the expertise of gastroenterologists to identify diseases. However, it is a subjective process, and the interpretation can vary even between expert clinicians. Considering recent progress in classifying gastrointestinal anomalies and landmarks in endoscopic and video capsule endoscopy images, this study proposes a hybrid model incorporating the advantages of Transformers and Convolutional Neural Networks (CNNs) for enhanced classification performance. Our model utilizes DenseNet201 as a CNN branch to extract local features and integrates the Swin Transformer branch for global feature understanding. Both of their features are combined to perform the classification task. For the GastroVision dataset, our proposed model demonstrates excellent performance with Precision, Recall, F1 score, Accuracy, and Matthews Correlation Coefficient (MCC) of 0.8320, 0.8386, 0.8324, 0.8386, and 0.8191, respectively, showcasing its robustness against class imbalance dataset and surpassing other CNNs as well as Swin Transformer model. Similarly, for the Kvasir-Capsule, a large video capsule endoscopy dataset, our model surpassed all other models, thereby achieving overall Precision, Recall, F1 score, Accuracy, and MCC of 0.7007, 0.7239, 0.6900, 0.7239, and 0.3871. Moreover, we generated saliency maps to explain our model's focus areas, showing its reliable decision-making process. The results underscore the potential of our hybrid CNN-Transformer model in aiding the early and accurate detection of gastrointestinal (GI) anomalies.

Keywords: Swin Transformer · Deep learning · Image classification · Gastrointestinal tract · GastroVision · Kvasir-Capsule

## 1 Introduction

Medical image analysis has become increasingly essential in the diagnosis and prognosis of numerous medical conditions. One of the leading causes of cancer death is colorectal cancer [22]. It often begins as a growth called a polyp inside the colon or rectum. While not every polyp evolves into cancer, early identification and removal can halt the progression to cancer. Detecting these conditions early through screening significantly boosts survival chances, since it facilitates intervention during earlier, more treatable stages of the diseases. However, diagnosing these diseases is very time-consuming and tedious. Therefore, diagnostic tools enhanced with Artificial Intelligence (AI) have been developed lately to aid healthcare professionals in more effectively identifying and addressing these health issues. This approach also improves the quality of medical image analysis and reduces the time and resources needed for proper diagnosis, thereby improving the overall efficiency in healthcare sectors.

The advancement in deep learning algorithms, especially CNNs, which include blocks of convolutional layers, pooling layers, and fully connected layers, has significantly impacted computer vision tasks. Acting as foundational networks, CNNs have demonstrated remarkable effectiveness across various computer vision tasks, such as image classification [14], object detection [18], and image segmentation [4]. CNNs are inherently predisposed to identify patterns in data, providing flexible and adaptive operations that can capture the specific characteristics of the data. They are also effective when used to analyze images obtained from endoscopy [3], leading to more accurate and efficient diagnoses. Recent advancements in deep learning have witnessed the adaptation of transformers, originally designed for natural language processing tasks, to the domain of image analysis. A notable example of this transformative paradigm shift is the Swin Transformer, which demonstrates considerable potential for image classification tasks [15]. It offers a hierarchical vision by using shifted windows, enabling it to capture both global and local features of images.

This study presents an innovative approach to endoscopic image classification that combines the strengths of both CNNs and Swin Transformers. The proposed model, which uses a DenseNet201 [12] as the CNN branch and the Swin Transformer as the other branch, can efficiently classify endoscopic images. Our extensive experiments demonstrate the model's robustness and superior performance on the two medical GI datasets, paving the way for its application in clinical workflows.

The main contribution of our work is as follows:

- 1. We have presented a hybrid architecture for a gastrointestinal (GI) image classification task that effectively combines the capabilities of CNNs and Swin Transformer, bringing a new approach to GI image analysis.
- 2. We interpreted and visualized the performance of our models using saliency maps. This approach provides in-depth insights into the model's decisionmaking process about which parts of the input data are most influential in the model's predictions.
- 3. Our Hybrid model is able to achieve the MCC of 0.8191 for the GastroVision dataset [13] (highly imbalanced data with 22 classes) and MCC of 0.3871 for the Kvasir-Capsule dataset [24] (imbalance data with 11 classes), outperforming the standalone DenseNet201, Swin-T and other CNN methods across all evaluation metrics.

## 2 Related Work

Our work relates to CNN, Transformer, and GI tract diseases and findings. Here, we thoroughly review the literature that bears significant relevance to these areas.

### 2.1 Gastrointestinal disease

Recent studies on the classification of endoscopic images have proposed several deep learning-based techniques [\[5,](#page-9-6) [7,](#page-9-7) [16\]](#page-10-3). For instance, Gamage et al. [9] presented the implementation of aggregation of deep learning features to predict anomalies related to digestive tract diseases. The ensemble involves the use of pre-trained CNN models such as DenseNet-201 [12], ResNet-18 [11], and VGG-16 [23] as the core feature extractors, integrated with a Global Average Pooling (GAP) layer. Similarly, Thambawita et al. [28] presented five different methods for GI tract disease classification task. Their combination of Densenet-161 and Resnet-152 with an additional MLP showcased highest performance. Afriyie et al. [2] presented Dn-CapsNets for identifying gastrointestinal tract diseases in the Kvasir-v2 dataset [19]. Srivastava et al. [25] introduced FocalConvNet, a network that merges focal modulation with lightweight convolutional layers, for the classification of anatomical and luminal findings within the Kvasir-Capsule dataset.

### 2.2 Transformer network

Recently, architectures based on transformers have become popular due to their proficiency in handling long-term dependencies. The transformer architecture, which is commonly utilized for natural language processing, has been shown by Dosovitskiy et al. [8] to be effective when employed for computer vision tasks as well. According to their research, Vision Transformer (ViT) performs exceptionally well on image classification tasks when applied to sequences of image patches. Similarly, Touvron et al. [29] developed a novel teacher-student strategy for training convolution free transformers on ImageNet, using a distillation token to facilitate learning from the teacher through attention. Similarly, Usman et al. [30] and Matsoukas et al. [17] compared the performance of Transformers and CNN in various image classification tasks. Likewise, Tang et. al [27] proposed a Transformer-based Multi-task Network to analyze GI-tract lesions automatically by combining the benefits of both the CNN and Transformer.

## 3 Methodology

Here, we merged the capabilities of CNN and transformer networks to address the challenge of identifying the images encountered in endoscopic procedures. Our model has three components - CNN branch, Transformer branch, and Classifier block as shown in Figure [1.](#page-3-0) The CNN branch captures detailed, local features while the Transformer branch focuses on broader, global aspects. These are subsequently merged into a single feature vector. This concatenated feature vector

![](assets/figures/_page_3_Figure_2.jpeg)

Fig. 1: Schematic representation of the Hybrid CNN-Transformer model.

is fed into the *Classifier block* for the final classification task. The dense block in the figure consists of several layers, including a dense layer with 256 units, followed by a LeakyReLU activation with an alpha of 0.1, batch normalization, and a dropout layer with a 0.5 rate.

### 3.1 Transformer Branch

For the transformer branch, we utilized a variant of Swin Transformer, namely Swin-T, where 'T' refers to the tiny. The model was initialized with weights pre-trained on the ImageNet1K dataset [20]. Here, an input RGB image is first divided into non-overlapping patches of  $4\times 4$ , each patch serving as a token. These patches have a feature dimension of  $4\times 4\times 3=48$ . Each patch is then linearly transformed into an embedding. The Swin Transformer model [15] consists of multiple layers of transformer encoders. Notably, the Swin Transformer replaces the traditional multi-head self-attention module in a Vision Transformer block with a shifting window-based self-attention. This design allows the model to attend to nearby patches while focusing on local context, without needing large attention windows. After processing the embeddings through several transformer layers, the information from various patches is aggregated to produce a single feature vector.

### 3.2 CNN Branch

In this branch, a CNN is used, specifically DenseNet201. We fine-tuned all the layers of the DenseNet201 on our dataset to enable it to extract features pertinent to our specific classification task. These densenet layers are then followed by Dense block.

![](assets/figures/_page_4_Figure_2.jpeg)

Fig. 2: Example samples from GastroVision [13] and Kvasir-Capsule [24]. The first three columns display images from GI endoscopy (GastroVision dataset), last three columns display from video capsule endoscopy (Kvasir-Capsule).

## 4 Dataset

We employed two multi-class publicly available GI endoscopy datasets: Gastro-Vision and the Kvasir-Capsule dataset. Some sample images of the datasets can be observed from the Figure [2.](#page-4-0)

- 1. GastroVision [13]: The GastroVision dataset encompasses a wide variety of classes, including anatomical landmarks, pathological findings, cases of polyp removal, and normal or regular findings. A total of 7,930 images from 22 classes were used in the experiment following the dataset provider.
- 2. Kvasir-Capsule [24]: Kvasir-Capsule is the largest publicly available video capsule endoscopy dataset, which comprises 44,228 meticulously labeled images representing 13 distinct classes of anatomical and luminal findings. We have utilized 11 out of 13 classes, as some classes have very few samples.

## 5 Experiments

### 5.1 Implementation details

Pre-Training : In this study, all the models were implemented using Tensorflow [1] framework. We then resized all images to a standard dimension of 224 × 224 pixels and performed normalization and different data augmentation techniques to mitigate the data limitations. As part of the data augmentation process, we applied several techniques from TensorFlow's Keras utilities, including rescaling pixel intensities, inducing shear transformations, applying rotations of up to 30 degrees, and performing vertical flips. Additionally, we adjusted the brightness to account for variations in lighting conditions. These strategic preprocessing and augmentation measures diversified our dataset and enhanced the generalization capabilities of our implemented architectures.

| Method           | Precision | Recall | F1-score                                                                        | Accuracy | MCC |
|------------------|-----------|--------|---------------------------------------------------------------------------------|----------|-----|
| MobileNetV2 [21] |           |        | 0.7308 ± 0.0356 0.7400 ± 0.0306 0.7318 ± 0.0332 0.7400 ± 0.0306 0.7083 ± 0.0347 |          |     |
| ResNet50 [10]    |           |        | 0.7151 ± 0.0154 0.7320 ± 0.0159 0.7170 ± 0.0179 0.7320 ± 0.0159 0.6988 ± 0.0182 |          |     |
| Xception [6]     |           |        | 0.7410 ± 0.0050 0.7499 ± 0.0032 0.7430 ± 0.0040 0.7499 ± 0.0032 0.7195 ± 0.0036 |          |     |
| InceptionV3 [26] |           |        | 0.7756 ± 0.0070 0.7847 ± 0.0049 0.7774 ± 0.0053 0.7860 ± 0.0048 0.7600 ± 0.0054 |          |     |
| Densenet201 [12] |           |        | 0.8056 ± 0.0062 0.8112 ± 0.0052 0.8046 ± 0.0056 0.8112 ± 0.0052 0.7886 ± 0.0059 |          |     |
| Swin-T [15]      |           |        | 0.8075 ± 0.0023 0.8148 ± 0.0038 0.8082 ± 0.0031 0.8148 ± 0.0038 0.7924 ± 0.0042 |          |     |
| DenseNet201 [12] |           |        |                                                                                 |          |     |
| + Swin-T [15]    |           |        | 0.8320 ± 0.0204 0.8386 ± 0.0221 0.8324 ± 0.0250 0.8386 ± 0.0035 0.8191 ± 0.0038 |          |     |

Table 1: Quantitative results on the GastroVision [13] dataset.

Experiment setup and configuration: Our experiments revolve around the task of classifying images using various architectures. To optimize our model, we used a trial-and-error approach to adjust various hyperparameters, such as learning rate, batch size, loss function, and optimizer. Specifically, the initial learning rate was set to 1e −2 for the GastroVision dataset and 1e −3 for the Kvasir-Capsule dataset, with dynamic adjustments made using the ReduceL-ROnPlateau scheduler to optimize convergence. We utilized categorical crossentropy as the loss function, which is suitable for multi-class classification tasks. These hyperparameters were iteratively refined through multiple experimental runs to achieve the best performance metrics for each dataset. The GastroVision dataset is divided into an 80:20 ratio for training and testing, respectively. Similarly, we used [official split](https://github.com/simula/kvasir-capsule/tree/master/official_splits) (split1) for Kvasir-Capsule dataset.

### 5.2 Evaluation metrics

We used various standard computer vision metrics to evaluate the performance of the models. These include MCC, weighted average precision, F1-score, recall, and overall accuracy. Of all these metrics, we prioritized MCC as the key metric since it is a more reliable statistical measure that yields a high score only when the classifier accurately predicts the majority of positive and negative data instances, and when both positive and negative predictions are mostly correct. We calculated these metrics over stratified 5-fold cross-validation and presented their averages along with their standard deviations. Additionally, we plotted saliency maps to visualize and interpret the performance of our model.

## 6 Results

We examined the trained model's effectiveness through multiple quantitative measures across a 5-fold cross-validation and the result is presented in Table [1](#page-5-0) and Table [2.](#page-6-0) Our hybrid CNN-Transformer model's performance is compared against the standalone DenseNet201, Swin Transformer model and other CNN methods. Here, each entry in the table is expressed as the mean ± SD of the respective performance metric, calculated over the 5-folds of cross-validation. We have reported the standard deviation (SD) to show the consistency of the model performance across five different folds, where a lower SD indicates consistent performance.

| Method           | Precision | Recall                                                                          | F1-score | Accuracy | MCC |
|------------------|-----------|---------------------------------------------------------------------------------|----------|----------|-----|
| MobileNetV2 [21] |           | 0.6100 ± 0.0047 0.6681 ± 0.0029 0.6283 ± 0.0028 0.6681 ± 0.0029 0.2489 ± 0.0074 |          |          |     |
| ResNet50 [10]    |           | 0.6257 ± 0.0020 0.6810 ± 0.0026 0.6402 ± 0.0020 0.6810 ± 0.0026 0.2785 ± 0.0055 |          |          |     |
| Xception [6]     |           | 0.5972 ± 0.0026 0.6857 ± 0.0012 0.6173 ± 0.0014 0.6857 ± 0.0012 0.2186 ± 0.0039 |          |          |     |
| InceptionV3 [26] |           | 0.5964 ± 0.0040 0.6861 ± 0.0019 0.6207 ± 0.0027 0.6861 ± 0.0019 0.2258 ± 0.0060 |          |          |     |
| Densenet201 [12] |           | 0.6726 ± 0.0041 0.7043 ± 0.0044 0.6711 ± 0.0035 0.7043 ± 0.0044 0.3508 ± 0.0067 |          |          |     |
| Swin-T [15]      |           | 0.6951 ± 0.0081 0.7088 ± 0.0068 0.6785 ± 0.0082 0.7088 ± 0.0068 0.3600 ± 0.0173 |          |          |     |
| DenseNet201 [12] |           |                                                                                 |          |          |     |
| + Swin-T [15]    |           | 0.7007 ± 0.0238 0.7239 ± 0.0105 0.6900 ± 0.0180 0.7239 ± 0.0105 0.3871 ± 0.0333 |          |          |     |

Table 2: Quantitative results on the Kvasir-Capsule [24] dataset.

The classification performance of the models trained on GastroVision dataset is shown in Table [1.](#page-5-0) For this dataset, our proposed hybrid model, the combined Swin Transformer and DenseNet201, consistently outperformed the other models across all performance metrics. With a F1 score, accuracy, and MCC of 0.8324, 0.8386 and 0.8191 respectively, our model demonstrated the highest performance. Moreover, the table reveals that MCC of the hybrid model surpasses pretrained CNN methods by over 3% and exceeds Swin T by 2%. Most competitive to the Hybrid model is Swin T, with MCC of 0.7924. Moreover, the classification performance of various models trained on the Kvasir-Capsule dataset is presented in Table [2.](#page-6-0) The hybrid model achieved F1-score, Accuracy, and MCC of 0.6900, 0.7239, and 0.3871, respectively, outperforming all CNN-based methods and the standalone Swin Transformer model. Furthermore, as depicted in table, the MCC of the hybrid model trained on Kvasir-Capsule dataset outperforms pretrained CNN methods by more than 3% and surpasses Swin T by 2%. Swin T emerges as the closest competitor to the hybrid model. The disparity in the classification performance in two datasets can be attributed to the characteristics and quality of the datasets. The GastroVision dataset, with 22 classes, likely benefits from higher quality images and more distinct class features. In contrast, the Kvasir-Capsule dataset, despite its larger size, has significant class imbalance which poses a challenge for effective model training and performance.

## 7 Discussion

### 7.1 Limitations and open challenges

We encountered several problems, including data limitations and problems with resource usage during the study. Moreover, the inherent imbalance within the datasets added complexity to the training process, requiring careful handling to prevent biased model outcomes. We tried solving the problem of data limitations using various data augmentation techniques. To date, we have not yet engaged with clinicians to incorporate their invaluable expertise. Recognizing their critical role, we see a significant opportunity to collaborate with them to further refine our model's architecture and enhance its clinical relevance. Moving forward, our aim is to foster interdisciplinary collaboration, paving the way for the development of more effective and clinically applicable AI solutions.

![](assets/figures/_page_7_Figure_2.jpeg)

Fig. 3: Saliency maps visualization for GastroVision dataset for four representative classes. The Figure shows the significant regions in the input image that contribute to the model's decision.

### 7.2 Visualization using saliency maps

A saliency map is a popular technique for visualizing the areas in the image that the model prioritized as the most important when making a prediction. To understand and interpret the behavior of our model, we have generated saliency maps for different representative classes, as shown in Figure [3.](#page-7-0) These saliency maps are produced by computing the gradients of the model's output with respect to its input image, highlighting the pixels that have the most significant impact on the model's decision-making process.

In the figure, each class consists of four panels: the original image, the saliency map highlighting the regions with the highest influence on the model's decision, the overlay of the saliency map on the original image, and a binary mask derived from the saliency map. This comprehensive visualization helps in interpreting the model's focus areas, further strengthening its reliability and trustworthiness, which are essential in high-risk medical applications, such as GI disease classification tasks. For instance, in the "Colon Polyps" example, the saliency map emphasizes the area where the polyp is located, indicating that the model has learned to focus on the key feature relevant to diagnosing polyps. Similarly, for the "Dyed Lifted Polyps" class, the saliency map shows a strong focus on the dyed region, which is crucial for identifying the presence of a lifted polyp. These focused regions in the saliency maps correspond to the medically significant features that are critical for accurate classification, thereby demonstrating the model's interpretability. Additionally, in the "Accessory Tools" class, the saliency map highlights the regions where the medical tools are present in the endoscopic images. This level of interpretability is similarly applicable to other classes in the dataset, where the saliency maps consistently highlight the most relevant and significant features required for accurate classification.

By overlaying the saliency map on the original image, as depicted in Figure [3,](#page-7-0) we can see a clear distinction between relevant and irrelevant areas. By providing a visual representation of what the model deems important, saliency maps reveal whether the model is focusing on appropriate features. This is particularly important in medical imaging, where the accuracy and reliability of the model's focus are critical for patient outcomes.

## 8 Conclusion

In this work, we proposed a hybrid model for GI image classification that combines the advantages of both CNN and Transformer for improved classification performance. The performance of our proposed models was evaluated using stratified 5-fold cross-validation and compared against individual DenseNet201, Swin Transformer model and other CNN methods. Our experimental results demonstrated that the proposed hybrid model consistently outperformed other methods, achieving an MCC of 0.8191 for the GastroVision dataset and an MCC of 0.3871 for the Kvasir-Capsule dataset. Furthermore, we employed saliency maps to visualize and interpret the decision-making process of our models. Additionally, the success of our model in classifying endoscopic images opens up new possibilities for its application in other medical imaging sectors, potentially enhancing clinical decision-making and improving patient well-being.

Acknowledgements This project is supported by NIH funding: R01-CA246704, R01-CA240639, U01-DK127384-02S1, and U01-CA268808.

### References

- 1. Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, A., Dean, J., Devin, M., et al.: Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467 (2016)
- 2. Afriyie, Y., A. Weyori, B., A. Opoku, A.: Gastrointestinal tract disease recognition based on denoising capsule network. Cogent Engineering 9(1), 2142072 (2022)
- 3. Ahmed, A.: Classification of gastrointestinal images based on transfer learning and denoising convolutional neural networks. In: Proceedings of International Conference on Data Science and Applications: ICDSA 2021, Volume 1. pp. 631–639 (2022)
- 4. Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M., Asari, V.K.: Recurrent residual convolutional neural network based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955 (2018)
- 5. Chang, Y.Y., Li, P.C., Chang, R.F., Yao, C.D., Chen, Y.Y., Chang, W.Y., Yen, H.H.: Deep learning-based endoscopic anatomy classification: an accelerated approach for data preparation and model validation. Surgical Endoscopy pp. 1–11 (2021)
- 6. Chollet, F.: Xception: Deep learning with depthwise separable convolutions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1251–1258 (2017)
- 7. Chou, C.K., Nguyen, H.T., Wang, Y.K., Chen, T.H., Wu, I.C., Huang, C.W., Wang, H.C.: Preparing well for esophageal endoscopic detection using a hybrid model and transfer learning. Cancers 15(15), 3783 (2023)
- 8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)
- 9. Gamage, C., Wijesinghe, I., Chitraranjan, C., Perera, I.: Gi-net: anomalies classification in gastrointestinal tract through endoscopic imagery with deep learning. In: 2019 Moratuwa Engineering Research Conference (MERCon). pp. 66–71 (2019)
- 10. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)
- 11. He, K., Zhang, X., Ren, S., Sun, J.: Identity mappings in deep residual networks. In: Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14. pp. 630–645 (2016)
- 12. Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4700–4708 (2017)
- 13. Jha, D., Sharma, V., Dasu, N., Tomar, N.K., Hicks, S., Bhuyan, M., Das, P.K., Riegler, M.A., Halvorsen, P., de Lange, T., et al.: Gastrovision: A multi-class endoscopy image dataset for computer aided gastrointestinal disease detection. arXiv preprint arXiv:2307.08140 (2023)
- 14. Li, Q., Cai, W., Wang, X., Zhou, Y., Feng, D.D., Chen, M.: Medical image classification with convolutional neural network. In: 2014 13th international conference on control automation robotics & vision (ICARCV). pp. 844–848 (2014)
- 15. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2021)

- 16. Lopez-Tiro, F., Flores-Araiza, D., Betancur-Rengifo, J.P., Reyes-Amezcua, I., Hubert, J., Ochoa-Ruiz, G., Daul, C.: Boosting kidney stone identification in endoscopic images using two-step transfer learning. In: Mexican International Conference on Artificial Intelligence. pp. 131–141. Springer (2023)
- 17. Matsoukas, C., Haslum, J.F., S¨oderberg, M., Smith, K.: Is it time to replace cnns with transformers for medical images? arXiv preprint arXiv:2108.09038 (2021)
- 18. Papageorgiou, C.P., Oren, M., Poggio, T.: A general framework for object detection. In: Sixth international conference on computer vision (IEEE Cat. No. 98CH36271). pp. 555–562 (1998)
- 19. Pogorelov, K., Randel, K.R., Griwodz, C., Eskeland, S.L., de Lange, T., Johansen, D., Spampinato, C., Dang-Nguyen, D.T., Lux, M., Schmidt, P.T., Riegler, M., Halvorsen, P.: Kvasir: A multi-class image dataset for computer aided gastrointestinal disease detection. In: Proceedings of the 8th ACM on Multimedia Systems Conference. pp. 164–169 (2017)
- 20. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al.: Imagenet large scale visual recognition challenge. International journal of computer vision 115, 211–252 (2015)
- 21. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4510–4520 (2018)
- 22. Siegel, R.L., Giaquinto, A.N., Jemal, A.: Cancer statistics, 2024. CA Cancer J Clin 74(1), 12–49 (2024)
- 23. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)
- 24. Smedsrud, P.H., Thambawita, V., Hicks, S.A., Gjestang, H., Nedrejord, O.O., Næss, E., Borgli, H., Jha, D., Berstad, T.J.D., Eskeland, S.L., et al.: Kvasir-capsule, a video capsule endoscopy dataset. Scientific Data 8(1), 142 (2021)
- 25. Srivastava, A., Tomar, N.K., Bagci, U., Jha, D.: Video capsule endoscopy classification using focal modulation guided convolutional neural network. In: Proceedings of the IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS). pp. 323–328 (2022)
- 26. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2818–2826 (2016)
- 27. Tang, S., Yu, X., Cheang, C.F., Liang, Y., Zhao, P., Yu, H.H., Choi, I.C.: Transformer-based multi-task learning for classification and segmentation of gastrointestinal tract endoscopic images. Computers in Biology and Medicine 157, 106723 (2023)
- 28. Thambawita, V., Jha, D., Riegler, M., Halvorsen, P., Hammer, H.L., Johansen, H.D., Johansen, D.: The medico-task 2018: Disease detection in the gastrointestinal tract using global features and deep learning. In: Proceedigns of the Medico 2018 (2018)
- 29. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., J´egou, H.: Training data-efficient image transformers & distillation through attention. In: Proceedings of the International Conference on Machine Learning. pp. 10347–10357 (2021)
- 30. Usman, M., Zia, T., Tariq, A.: Analyzing transfer learning of vision transformers for interpreting chest radiography. Journal of digital imaging 35(6), 1445–1462 (2022)