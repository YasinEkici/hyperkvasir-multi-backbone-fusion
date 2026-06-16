# Vision Transformer With Hybrid Shifted Windows for Gastrointestinal Endoscopy Image Classification

Wei Wang, Xin Yan[g](https://orcid.org/0000-0001-6252-1061) , *Member, IEEE*, and Jinhui Tan[g](https://orcid.org/0000-0001-9008-222X) , *Senior Member, IEEE*

*Abstract*— Automated classification of gastrointestinal endoscope images can help reduce the workload of doctors and improve the accuracy of diagnoses. The rapidly developed vision Transformer, represented by Swin Transformer, has become an impressive technique for medical image classification. However, Swin Transformer cannot capture the long-range dependency well in complex gastrointestinal endoscopy images. As a result, it fails to represent features of some widely-spread targets in digestive tract images, such as normal-z-line and esophagitis, effectively. To solve this problem, we propose a novel vision Transformer model based on hybrid shifted windows for digestive tract image classification, which can obtain both short-range and long-range dependency concurrently. Extensive experiments demonstrate the superiority of our method to the state-of-theart methods with a classification accuracy of 95.42% on the Kvasir v2 dataset and a classification accuracy of 86.81% on the HyperKvasir dataset.

*Index Terms*— Image classification, vision transformer, computer aided diagnosis, gastrointestinal endoscope.

## I. INTRODUCTION

D IGESTIVE tract diseases can seriously affect people's quality of life. In recent years, the incidence of digestive tract diseases has increased rapidly in Asia [\[1\], \[](#page-7-0)[2\]. As](#page-7-1) the only way to directly view the mucous lining of a digestive tract, gastrointestinal endoscopy has become an irreplaceable diagnostic tool in gastrointestinal examinations. However, manual interpretation of endoscopy images is time-consuming and error-prone. Therefore, automatically classifying gastrointestinal endoscopy images with lesion regions from normal ones is in high demand.

Over the past decade, convolutional neural networks (CNNs) have achieved great success in computer vision and have become the mainstream methods for medical image

Manuscript received 1 February 2023; revised 17 April 2023; accepted 12 May 2023. Date of publication 25 May 2023; date of current version 6 September 2023. This work was supported in part by the National Natural Science Foundation of China under Grant 62061160490, Grant 62122029, and Grant U20B2064. This article was recommended by Associate Editor J. Wu. *(Corresponding author: Xin Yang.)*

Wei Wang and Jinhui Tang are with the School of Computer Science and Engineering, Nanjing University of Science and Technology, Nanjing, Jiangsu 210094, China (e-mail: weiwang@njust.edu.cn; jinhuitang@njust.edu.cn).

Xin Yang is with the School of Electronic Information and Communications, Huazhong University of Science and Technology, Wuhan, Hubei 430074, China (e-mail: xinyang2014@hust.edu.cn).

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TCSVT.2023.3277462.

Digital Object Identifier 10.1109/TCSVT.2023.3277462

classification. Due to the characteristics of digestive tract diseases, the focal region could either take up a very small area in an endoscopic image or spread along a certain direction. However, most CNN-based classifiers have two inherent defects that limit their ability to achieve satisfactory accuracy for classifying gastrointestinal endoscopy images. Firstly, CNNs lack built-in saliency mechanisms, which could misclassify images with small lesions. Secondly, CNNs lack the ability to learn long-range dependence, which leads to poor classification performance for images with long focal regions, such as normal z-line and esophagitis.

The recently developed Vision Transformer (ViT) model [3] provides potential in addressing the aforementioned limitations. ViT takes patches of fixed size as input which are flattened, converted into vectors by the embedding layer, and then fed into the Transformer layer. The Transformer layer computes self-/cross-attention between all pairs of vectors and learns long-distance dependency. Despite the advantages of ViT, its performance in classifying complex images is still inferior to the state-of-the-art CNN-based classifers due to the following two reasons: 1) ViT, which is based on global self-attention, is weak in capturing local context, leading to poor classification of categories containing small targets. 2) Compared with CNNs, ViT lacks inductive bias and therefore requires pre-training on very large datasets (such as JFT-300M) to achieve competitive results.

To solve these problems, Swin Transformer [\[4\] pro](#page-7-3)poses to compute local self-attention instead of global self-attention and adopts shifted windows to capture interactions between different windows to learn long-distance dependency. Moreover, Swin Transformer imitates the hierarchical structure of CNNs by dividing the model into four stages, each of which consists of a certain number of Transformer blocks. The feature maps of the Transformer blocks have the same size inside each stage and reduced size in the subsequent stages. Different stages are connected by a patch merging layer.

However, the appearance of landmarks and lesions at different sites of digestive tract varies greatly which limits the ability of Swin Transformer with a single shifted window pattern. Figure [1](#page-2-0) shows exemplar images of the Kvasir v2 dataset used in our experiments, which contains 3 types of disease images and 3 types of healthy digestive tract landmark images. It can be seen that the focal area of some types of images concentrates in a local part of the image, but the other types

1051-8215 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

of images, e.g., normal-z-line and esophagitis, are exceptions. The classification of these two types of images is based on the morphology of z-line, which is a regular dividing line in the human esophagus and is used as a diagnostic marker for some diseases in endoscopic diagnosis. Esophagitis can cause the z-line to become blurred. Different from the focal areas of other diseases, z-line is usually distributed in a large area of the image, so it can not be completely described by local features. In order to well describe features of both small focal regions and widely-spreading targets, we proposed a vision Transformer based on hybrid sifted windows, named HSW Transformer. Our HSW Transformer leverages half of the multi-heads to calculate self-attention in standard shifted windows. The remaining half of the multi-heads are used to calculate the self-attention based on long rectangular windows to better represent features of widely-spreading targets with long-range dependency. Through this strategy, our model not only retains the strong ability of capturing multi-scale features and keeping low computational complexity compared with ViT, but also enhances the learning ability of long-distance dependence for gastrointestinal endoscopy images.

To summarize, the main contributions of this paper include:

- 1) A novel vision Transformer based on hybrid shifted windows for accurate and efficient gastrointestinal endoscopy image classification. Our model can well capture features of small focal regions and widelyspreading targets in gastrointestinal endoscopy images.
- 2) To the best of our knowledge, we are the first to use the vision Transformer for endoscopic image classification. Experimental results on the Kvasir v2 datasets demonstrate the superior performance of our method to the state-of-the-art methods.

The rest of the paper is organized as follows. Section [II](#page-1-0) introduces the related works about computer-aided gastrointestinal endoscopic image classification and vision Transformer. Section [III](#page-3-0) presents details of our method. Experimental results and analysis are provided in section [IV.](#page-5-0) Section [V](#page-7-4) concludes the paper.

# II. RELATED WORKS

## A. Gastrointestinal Endoscopic Image Classification

Computer-aided diagnosis of gastrointestinal endoscopic images has been studied for several decades. The earliest research dates back to 1998 when Krishnan et al. [5] proposed to distinguish abnormal endoscopic images from normal ones based on the curvature of contours. In [\[6\],](#page-8-0) Dhandra et al. designed a morphological watershed segmentation method to classify gastrointestinal endoscopic images into normal and abnormal ones according to the segmentation results. Wang et al. [\[7\] in](#page-8-1)troduced LBP features to represent endoscopic images and used a log-likelihood ratio to measure the similarity of features in different regions. Similarly, Magoulas et al. [\[8\] use](#page-8-2)d a combination of concurrence matrices and 2D discrete wavelet transform for image feature representation and the neural network to classify endoscopic images. Kodogiannis et al. [\[9\] pro](#page-8-3)posed an advanced fuzzy inference neural network to classify endoscopic images. Lima et al. [\[10\] u](#page-8-4)tilized the color wavelet covariance features and the radial basis function-based classifier to detect abnormal lesions in endoscopic images.

With the rapid development of deep learning, CNNbased classifiers such as VGG16 [\[11\],](#page-8-5) ResNet [12] and DenseNet [\[13\] h](#page-8-7)ave achieved promising results and become the mainstream image classification tools. Some researchers [\[14\], \[](#page-8-8)[15\],](#page-8-9) [\[16\] h](#page-8-10)ave applied a variety of CNN models pre-trained on the ImageNet dataset to classify endoscopic images. For instance, Shichijo et al. [\[17\] a](#page-8-11)pplied a 22-layer CNN model for the diagnosis of Helicobacter pylori infection. Georgakopoulos et al. [\[18\] e](#page-8-12)mployed a weaklysupervised CNN architecture to detect inflammatory gastrointestinal lesions. Hirasawa et al. [\[19\] b](#page-8-13)uilt a Single Shot MultiBox Detector based CNN to detect gastric cancer in endoscopic images. Iakovidis et al. [\[20\] p](#page-8-14)roposed a weakly supervised CNN framework to concurrently classify abnormality and locate abnormal areas. There are also several studies focusing on CNN-based polyp detection from the endoscopic images [\[21\], \[](#page-8-15)[22\], \[](#page-8-16)[23\], \[](#page-8-17)[24\], \[](#page-8-18)[25\]. W](#page-8-19)ang et al. [\[26\] p](#page-8-20)roposed a multi-task real-time deep neural network that can perform polyp detection, classification and segmentation. Specifically, this method utilized YOLOv4 [\[27\] fo](#page-8-21)r polyp detection, ResNet50-D [\[28\] fo](#page-8-22)r polyp classification and U-Net [\[29\] fo](#page-8-23)r polyp segmentation. Yuan et al. [\[30\] e](#page-8-24)mployed a densely connected neural network (DenseNet) to perform end-to-end polyp recognition. Then they introduced an unbalanced discriminant (UD) loss to deal with the severe data imbalance problem and a category sensitive (CS) loss to encourage interclass variations and reduce intra-class variations. Xing et al. [\[31\] p](#page-8-25)roposed the Attention Guided Deformation Network (AGDN) for the wireless capsule endoscopy image classification task. Their model is a two branch structure based on DenseNet. The attention maps learned by one branch are utilized to guide the deformation of input images of the other branch, yielding good classification performance of small lesions.

Several other studies leveraged a 2-stage classification pipeline (i.e., a CNN for feature extraction and an individual classifier for classification) rather than an end-to-end CNN model. Yu et al. [\[32\] u](#page-8-26)tilized the CNN as features extractor and the cascaded extreme learning machine (ELM) as the classifier. Zhu et al. [\[33\] c](#page-8-27)ombined a CNN feature extractor and the support vector machine (SVM) classifier to detect gastrointestinal diseases. Billah et al. [\[34\] e](#page-8-28)xtracted the color wavelet (CW) features and the CNN features of endoscopic images. These features are then combined and fed into SVM to detect the gastrointestinal polyps. Guo et al. [\[35\] u](#page-8-29)tilized EfficientNet [\[36\] fo](#page-8-30)r feature extraction based on weakly supervised learning. A new attention network consisting of a spatial attention module and a channel attention module is proposed as the classifier. The spatial attention maps are obtained by global average pooling, and the channel attention module is based on SENet [\[37\].](#page-8-31)

There are also several studies focusing on classifying anatomical landmarks of the digestive tract rather than detecting abnormal images and regions. Zou et al. [\[38\] p](#page-8-32)roposed a CNN-based framework to classify the digestive organs.

![](assets/figures/_page_2_Figure_2.jpeg)

Fig. 1. Examples from the Kvasir v2 image dataset which consists of 6 categories of endoscopic images. The 6 classes are: (a) esophagitis (b) polyps (c) ulcerative-colitis (d) normal-cecum (e) normal-pylorus (f) normal-z-line. The first 3 classes are pictures of different digestive tract diseases, and the last 3 classes are pictures of the health status of the corresponding areas. The two categories related to z-line morphology, i.e. esophagitis and normal-z-line are significantly different from the others.

Takiyama et al. [\[39\] u](#page-8-33)sed GoogLeNet [\[40\] to](#page-8-34) classify anatomical locations of esophagogastroduodenoscopy (EGD) images. Twinanda et al. [\[41\] p](#page-8-35)resented a multi-task CNN framework, called EndoNet, to classify surgical phases. Petscharnig et al. [\[42\] p](#page-8-36)resented a CNN model based on GoogLeNet to classify the gastrointestinal diseases and anatomical landmarks at the same time.

CNNs have strong ability to learn local features, but lack the ability of rotation invariance and learning long-range dependence. Such limitations make existing CNNs perform poorly in gastrointestinal endoscopic image classification. Wang et al. [\[43\] pr](#page-8-37)oposed a two-branch model based on a capsule network to address these limitations. However, the model based on the capsule network doesn't perform well on two classes of endoscopic images: normal-z-line and esophagitis.

## B. Vision Transformer Based Models

The Transformer structure was proposed by Vaswani et al. [\[44\] a](#page-8-38)nd was originally used to solve natural language processing (NLP) problems. Inspired by the Transformer's great success in NLP tasks, Dosovitskiy et al. [\[3\] ca](#page-7-2)me up with the Vision Transformer (ViT) model, which leverages the Transformer mechanics for computer vision tasks. In the framework of ViT, input images are split into patches of equal size, which are then transformed into patch embeddings by linear projection. Then, patch embeddings are fed into the Transformer layer to produce the final classification result. The ViT model has achieved comparable performance to CNNs on several classification tasks. However, ViT still has an obvious flaw: the lack of locality mechanisms, resulting in the inability to represent multi-scale features.

To address the limitations of ViT, Zhou et al. [\[45\] p](#page-8-39)roposed increasing the depth of ViT. However, deepening ViT yields quick performance saturation, and they analyze that the attention collapse issue is the cause of this phenomenon. As the number of layers increases, the feature maps between different layers become more and more similar. To alleviate this issue, Zhou et al. proposed a re-attention method to increase the attention diversity of different layers. Some researchers enhanced the ability of Transformer models to represent multi-scale features by introducing a hierarchical structure. Pyramid Vision Transformer (PVT) [\[46\] is](#page-8-40) the first attempt with a hierarchical design. Wang et al. introduced a pyramid structure similar to CNNs, which enables PVT to perform well in dense prediction tasks. Liu et al. [\[4\] pr](#page-7-3)oposed Swin Transformer, which adopts a hierarchical vision Transformer structure and computes feature representation within shifted windows. They split the model into four stages, each of which contains several Transformer blocks. Patch merging is used between different stages to change feature dimensions, resulting in a hierarchical representation similar to CNNs. Several studies focus on improving the attention module to get a better representation. For instance, Lin et al. [47] proposed Cross Attention, which calculates attention inside image patches and applies attention between image patches. Compared with the standard Transformer model ViT, this method can greatly reduce computational complexity and is able to capture multi-scale features. Wang et al. [\[48\] p](#page-8-42)roposed a Cross-scale Embedding Layer (CEL) and a Long Short Distance Attention (LSDA) module to reduce the time cost of ViT and capture multi-scale features. Chen et al. [49] proposed a pyramid structure and introduced a novel regionalto-local attention mechanism to replace global self-attention in vision Transformers. Other studies have combined convolution modules with Transformer modules to enhance capability in capturing local context. Wu et al. [\[50\] c](#page-8-44)ombined convolutions with vision Transformers and proposed convolutionbased token embedding and convolutional Transformer blocks. Their model achieves the state-of-the-art performance over ResNets on ImageNet-1K. Li et al. [\[51\] i](#page-8-45)ntroduced depthwise convolution into the vision Transformer-based model to enhance locality. CoAtNets [\[52\] le](#page-9-0)veraged vertically stacking convolution layers and attention layers in a principled way to introduce inductive bias. Wang et al. [\[53\] p](#page-9-1)roposed a model that takes into account both global and local features for the light field de-occlusion task. CNNs are utilized at shallow layers of the model to extract local features, and Transformers are employed at deep layers to obtain global representations. Wang et al. [\[54\] u](#page-9-2)tilized a pyramid structure based on CNN to extract local features and employed Vision Transformers to obtain image contextual information. They also proposed a global semantic NetVLAD aggregation strategy to aggregate multi-scale token maps of Transformer output.

# III. OUR METHOD

In this work, we improve Swin Transformer to achieve high classification accuracy on gastrointestinal endoscopy images. Due to the inherent characteristics of gastrointestinal endoscopy images, existing methods including Swin Transformer can hardly achieve satisfactory results for images with both small lesions and widely-spread lesions. Figure [1](#page-2-0) shows endoscopic images of three healthy landmarks and three diseases of the digestive tract. The appearances of z-line and esophagitis are quite different from those of other types of images. The focal areas of these two types of images are not limited to local areas but are distributed throughout the whole image. Previous studies about gastrointestinal endoscopy images classification have not realized that there are two distinct distribution types of focal areas, namely local distribution and global distribution. Existing models treat the focal areas of gastrointestinal endoscopy images as the same distribution, resulting in poor performance on normal z-line and esophagitis images.

In Section [IV-B,](#page-6-0) we tested the proportion of misclassified images in different categories of Kvasir v2 based on Swin Transformer. Transformer. The experimental results confirm our analysis and conjecture that the main classification errors of the existing models are in the two classes: normal-z-line and esophagitis. We analyze the poor performance of Swin Transformer on gastrointensitinal endoscopy images as being due to its leveraging of a hierarchical structure to capture multi-scale image information and improve training efficiency. However, such a strategy also reduces Swin Transformer's ability to capture global context information. As a result, Swin Transformer does not perform well on images with widely-spread focus regions. To better guide the model to learn long-range dependence without increasing computational consumption too much and maintaining its multi-scale feature learning ability, we design novel hybrid shifted windows and integrate them into Swin Transformer. As shown in Figure [4,](#page-5-1) the hybrid shifted windows consist of long rectangular-shaped shifted windows in horizontal and vertical directions and traditional shifted windows as in Swin Transformer. Such hybrid shifted windows enable us to capture images with both local-oriented distribution and global-oriented distribution. The experimental results in section [IV-B](#page-6-0) also demonstrate that our module can indeed improve the performance of the model on z-line and esophagitis images.

## A. Method Overview
The overall structure of the model is shown in Figure [2.](#page-4-0). It splits an input image with a size of $H \times W \times 3$ into nonoverlapping patches, like Swin Transformer. The patch size we used in our implementation is $4 \times 4$. Therefore,$H/4 \times W/4$ patches are obtained. Each patch is treated as a "token". Then a linear embedding layer is applied to transform the dimension of tokens into a specific value.
To obtain a hierarchical representation, we divide the network into four stages, each of which contains a set of Transformer blocks. In our implementation, we set the number of blocks to 2,2,6,2 for Stage 1 to 4, respectively. We use standard Swin Transformer blocks for the first two stages and Hybrid Swin Transformer blocks for the last two stages. The feature dimension remains unchanged within each stage. Patch merging between stages acts like the pooling layer, halving the feature dimensions. Patch merging is used to change the dimension of the feature between Transformer layers. Assuming the dimension of each token is C, patch merging merges the adjacent 2×2 tokens into a single feature with a dimension of 4C. Then a linear layer transforms the merged token to reduce its dimension to 2C. Through this mechanism, a hierarchical structure similar to CNNs is formed.

## B. Transformer Block Details

As shown in Figure [2,](#page-4-0) our model includes both standard Swin Transformer blocks and the newly proposed Hybrid Swin Transformer blocks.

*1) Swin Transformer Block:* The structure of the standard Swin Transformer block is shown in figure [3\(a\).](#page-4-1) In order to improve computational efficiency, Ze Liu et al. performed self-attention computation within local windows in the Swin Transformer. Such window-based self-attention lacks connections across windows, which limits its ability to capture global dependencies. To introduce cross-window dependencies while maintaining efficient computation, shifted window selfattention is applied after window-based attention. A standard Swin Transformer block consists of a window-based multihead self-attention (MSA) and a shifted window-based multihead self-attention layer, both blocks are followed by a 2-layer Multilayer Perceptron (MLP). Each MSA and MLP block is preceded by a LayerNorm (LN) layer. The standard Swin

![](assets/figures/_page_4_Figure_2.jpeg)

Fig. 2. The overall flow of our model. The whole model is divided into four stages, each of which contains several Transformer blocks.

![](assets/figures/_page_4_Figure_4.jpeg)

Fig. 3. The structure of Transformer blocks we used in our model: (a) standard Swin Transformer block (b) Hybrid Swin Transformer block.

Transformer blocks are computed as:

$$\hat{Z}^{l} = W - MSA(LN(Z^{l-1})) + Z^{l-1} \tag{1}$$

$$Z^{l} = MLP(LN(\hat{Z}^{l})) + \hat{Z}^{l} \tag{2}$$

$$\hat{Z}^{l+1} = SW - MSA(LN(Z^l)) + Z^l \tag{3}$$

$$Z^{l+1} = MLP(LN(\hat{Z}^{l+1})) + \hat{Z}^{l+1} \tag{4}$$

where *z $^{l}$* denotes the output feature of MLP for block *l* and *z*ˆ *$^{l}$* denotes the output feature of MSA for block *l*. W-MSA denotes the window-based multi-head self attention module. SW-MSA denotes the shifted window-based multi-head selfattention module.

*2) Hybrid Shifted Windows Block:* The structure of the Hybrid Swin Transformer block is shown in figure [3\(b\).](#page-4-1) There are two steps in each block: a standard window-based multi-head self-attention layer and a hybrid shifted windowsbased multi-head self-attention (HSW-MSA) layer. Figure [4](#page-5-1) shows our hybrid shifted windows consist of long rectangularshaped shifted windows in horizontal and vertical directions and traditional shifted windows as in the Swin Transformer. For each HSW-MSA, the input multi-heads are divided into three groups for different processing. Half of the multi-heads will calculate the standard shifted window-based self-attention. Half of the remaining multi-heads calculate the self-attention based on the horizontal strip windows, and the last quarter

![](assets/figures/_page_5_Figure_2.jpeg)

Fig. 4. (a) Partition of Windows in standard Swin Transformer blocks (b) Partition of Windows in Hybrid Swin Transformer blocks.

calculate the self-attention based on the vertical strip windows. As shown in figure [3\(b\),](#page-4-1) the hybrid Transformer blocks are computed as:

$$\hat{Z}^{l} = W - MSA(LN(Z^{l-1})) + Z^{l-1} \tag{5}$$

$$Z^{l} = MLP(LN(\hat{Z}^{l})) + \hat{Z}^{l} \tag{6}$$

$$\hat{Z}^{l+1} = HSW - MSA(LN(Z^l)) + Z^l \tag{7}$$

$$Z^{l+1} = MLP(LN(\hat{Z}^{l+1})) + \hat{Z}^{l+1} \tag{8}$$

where *z $^{l}$* denotes the output feature of MLP for block *l* and *z*ˆ *$^{l}$* denotes the output feature of MSA for block *l*. HSW-MSA denotes the hybrid shifted window-based selfattention module.

#### IV. EXPERIMENTS

## A. Dataset and Preprocessing

*1) Datasets:* We evaluate our method on the Kvasir v2 dataset [\[55\] a](#page-9-3)nd the HyperKvasir dataset [\[56\]. T](#page-9-4)he images in the Kvasir v2 dataset are representative frames extracted from endoscopic videos with resolutions ranging from 720×576 to 1920 × 1072 pixels. Each image in the dataset was annotated by experienced endoscopists to assess the effectiveness of computer-aided image classification and recognition methods. Kvasir v2 contains a total of 8 types of images, including 3 important anatomical landmarks, 3 digestive tract diseases, and 2 types of images related to endoscopic polypectomy. The three anatomical landmarks in the Kvasir v2 dataset are z-line, pylorus and cecum. The three categories of diseases in this dataset include esophagitis, polyps, and ulcerative colitis. The last two types of images related to polyp surgery are irrelevant to our task, so we did not use them in the experiments. There are 1000 pictures in each class. Figure [1](#page-2-0) shows exemplar endoscopic images of the 6 categories used in our experiments. As can be seen from Figure [1,](#page-2-0) endoscopic images of the digestive tract not only contain different lesions and landmarks, but also contain different disturbances, such as bubbles, masking caused by food residues and faeces, deformation caused by digestive peristalsis, and changes in the viewing angle caused by changes in the position of the endoscope. These disturbances pose great challenges to the endoscopic image classification tasks.

To demonstrate the universality of our method, we also conducted experiments on another dataset - HyperKvasir. The HyperKvasir dataset is a brand new dataset published by the Kvasir dataset team. However, it contains different image classes and images from Kvasir v2. HyperKvasir is probably the largest and most complex gastrointestinal endoscopic visual dataset available so far. We only used labeled images in HyperKvasir. There are two major differences between the HyperKvasir dataset and the Kvasir v2 dataset : 1) The Hyper-Kvasir dataset is unbalanced, that is, the number of images contained in different categories varies greatly. This presents a greater challenge to all image classification methods. 2) The number of image classes in these two datasets is different. Specifically, in HyperKvasir, some diseases, such as ulcerative colitis, are classified into different categories according to the degree of the disease. The HyperKvasir dataset consists of 40 types of images. In this work, we used only 18 classes related to anatomical landmarks and pathological findings in HyperKvasir.

![](assets/figures/_page_6_Figure_2.jpeg)

![](assets/figures/_page_6_Figure_3.jpeg)

![](assets/figures/_page_6_Figure_4.jpeg)

![](assets/figures/_page_6_Figure_5.jpeg)

Fig. 5. The proportion of classification errors in different classes of images by different models including: (a) VGG16 (b) DenseNet201 (c) Swin Transformer (d) our model.

For the Kvasir v2 dataset, 1200 images (200 images from each category) are randomly selected as testing images. 3600 images (600 images from each category) are randomly selected as the training set and the remaining 1200 images are used as the validation samples.

For the HyperKvasir dataset, 4019 images (60% of images of each class) are randomly selected as training images. 1342 images (20% of images of each class) are randomly selected as the testing images and the remaining 1342 images are used as the validation cases.

- *2) Preprocessing:* A small screen embedded in the lower left corner of some images in these two datasets is used to show the position of the endoscope in the digestive tract. These small patches are irrelevant with the image classification, but interfere with the final classification results. Thus, we remove these patches from the images by assigning all pixels in the patch to 0.
- *3) Evaluation Metrics:* We evaluate our method using two widely-used metrics, i.e. accuracy and F1-score. Accuracy and F1-score are defined as follows:

$$Accuracy = \frac{TP + TN}{TP + TN + FN + FP} \tag{9}$$

$$F1 - score = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} \tag{10}$$

where TP, TN, FP, FN are the numbers of true positives, true negatives, false positives, and false negatives, respectively.

## B. Percentage of Classification Errors on Different Categories of Images

We compared the proportion of classification errors of different models for each class of the Kvasir v2 dataset. This helps explain why existing models cannot further improve the performance of endoscopic image classification. As shown in Figure [5,](#page-6-1) the experimental results are presented as pie charts.

According to the experimental results shown in Figure [5,](#page-6-1) we found that most of the classification errors occurred in normal z-line and esophagitis. Images of these two classes are different forms of the z-line in healthy and diseased throat locations. It can be seen from the schematic diagram that the z-line region is an irregular ring occupying a large area of the image, which is different from other types of focal regions concentrated which reside in a local area. Experimental results verify our conjecture that models mainly based on local features can hardly recognize relevant regions in images. It is necessary to introduce long-range dependence into the Swin Transformer model to handle cases of these two categories. In addition, it can be seen from 5(d) that our method has improved the classification performance on these two types of images.

## C. Comparison With the State-of-the-Art Methods

We compared our method with existing state-of-the-art methods, including VGG16, DenseNet121, DenseNet201,

TABLE I RESULTS OF COMPARISON WITH THE EXISTING METHODS ON THE KVASIR V2 DATASET

| model                | Accuracy (%) | F1-score |
|----------------------|--------------|----------|
| VGG16 [11]           | 86.83        | 0.8681   |
| DenseNet121 [13]     | 84.25        | 0.8410   |
| DenseNet201 [13]     | 88.42        | 0.8839   |
| Swin Transformer [4] | 87.5         | 0.8748   |
| L-DenseNetCaps [43]  | 94.83        | 0.9482   |
| HSW (our model)      | 95.42        | 0.9541   |

TABLE II RESULTS OF COMPARISON WITH THE EXISTING METHODS ON THE HYPERKVASIR DATASET

| model                | Accuracy (%) |  |
|----------------------|--------------|--|
| VGG19 [11]           | 81.30        |  |
| DenseNet121 [13]     | 84.28        |  |
| DenseNet201 [13]     | 85.02        |  |
| Swin Transformer [4] | 86.07        |  |
| L-DenseNetCaps [43]  | 85.99        |  |
| HSW (our model)      | 86.81        |  |

which are representative CNN classification models, L-DensenetCaps, which is an advanced capsulenet-based model and Swin Transformer. In this experiment, the architecture hyper-parameters of these model variants are:

- 1) Swin Transformers: The channel number of the hidden layers in the first stage is 96, the layer numbers are 2,2,6,2 and the numbers of heads in each stage are 3, 6, 12, 24.
- 2) L-DenseNetCaps: The routing iteration times of the two capsule layers are 2 and 3 respectively.
- 3) HSW Transformers: The channel number of the hidden layers in the first stage is 96, the layer numbers are 2,2,6,2 and the numbers of heads in each stage are 6, 12, 24,48.

Table [I](#page-7-6) shows the experimental results on the Kvasir v2 dataset.

Our method outperforms all the comparison methods and achieves a 1.17% improvement compared with the the stateof-the-art method L-DenseNetCaps. We also compare our method with the state-of-the-art methods on HyperKvasir to demonstrate the universality of our method. Experimental results in Table [II](#page-7-7) show that our method still achieves the best accuracy (i.e., 86.81%).

In addition, we measured inference throughput of our model and swin transformer tiny using a GPU V100. While the classification accuracy is significantly improved, our model has similar inference throughput as Swin Transformer. The inference throughput of our model is 54.6 images/s, and the inference throughput of Swin Transformer tiny is 56.8 images/s.

TABLE III

RESULTS OF COMPARISON EXPERIMENT WHEN DIFFERENT STAGES BEING REPLACED WITH HYBRID SWIN TRANSFORMER BLOCKS ON THE KVASIR V2 DATASET

| Replacing blocks of             | stage 4 | stage 3 and 4 |
|---------------------------------|---------|---------------|
| Accuracy (%)                    | 94.75   | 95.42         |
| Error rate on normal z-line (%) | 6.5     | 4.5           |
| Error rate on esophagitis (%)   | 9.5     | 8             |

## D. Ablation Study

In our final model, we replaced the Swin Transformer blocks of the last two stages with our Hybrid Swin Transformer blocks. In this section, we examine the performance of our method when replacing SW-MSA with our HSW-MSA at different stages. So we did an experiment on the Kvasir v2 dataset comparing two scenarios: only replacing blocks in stage 4 and replacing blocks in stage 3 and 4.

Results in Table [III](#page-7-8) show that replacing blocks of the third and fourth stages produces better results than replacing only the last stage. At the same time, we found that the model with replaced blocks in stage 3 and stage 4 had lower error rates in the normal z-line and esophagitis images compared to the model with only the last stage replaced. This verifies our analysis in Chapter 3, which is that the ability of the model to acquire long distance dependence is enhanced by adding long rectangular shifted windows through Hybrid Swin Transformer blocks. Thus, by increasing the number of Hybrid Swin Transformer blocks, the classification errors in both normal z-line and esophagitis were reduced.

## V. CONCLUSION

In this paper, we proposed a Hybrid Shifted Windows based Transformer model, named HSW Transformer, for gastrointestinal endoscopy image classification. Our HSW Transformer can maintain the ability of Swin Transformer to obtain multiscale features and, at the same time, effectively overcome the limitation of Swin Transformer in capturing long-range dependence. Extensive experiments demonstrate superior performance of our method to the state-of-the-art methods. The proposed method can help doctors to interpret gastrointestinal endoscopy images, thus improving the accuracy and timeliness of diagnosis, and also reducing the economic burden of patients.

## REFERENCES

- [1] J. E. Everhart and C. E. Ruhl, "Burden of digestive diseases in the United States part I: Overall and upper gastrointestinal diseases," *Gastroenterology*, vol. 136, no. 2, pp. 376–386, Feb. 2009.
- [2] K. Goh, "Changing trends in gastrointestinal disease in the Asia–Pacific region," *J. Digestive Diseases*, vol. 8, no. 4, pp. 179–185, Nov. 2010.
- [3] A. Dosovitskiy et al., "An image is worth 16 × 16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent.*, 2020, pp. 1–11.
- [4] Z. Liu et al., "Swin transformer: Hierarchical vision transformer using shifted windows," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Oct. 2021, pp. 10012–10022.
- [5] S. Krishnan, X. Yang, K. Chan, S. Kumar, and P. Goh, "Intestinal abnormality detection from endoscopic images," in *Proc. Annu. Int. Conf. IEEE Eng. Med. Biol. Soc.*, vol. 2, 1998, pp. 895–898.

- [6] B. Dhandra and R. Hegadi, "Classification of abnormal endoscopic images using morphological watershed segmentation," in *Proc. Int. Conf. Cognit. Recognit.*, 2005, pp. 1–10.
- [7] P. Wang, S. M. Krishnan, C. Kugean, and M. P. Tjoa, "Classification of endoscopic images based on texture and neural network," in *Proc. Conf. Proc. 23rd Annu. Int. Conf. IEEE Eng. Med. Biol. Soc.*, 2001, pp. 3691–3695.
- [8] G. D. Magoulas, "Neuronal networks and textural descriptors for automated tissue classification in endoscopy," *Oncol. Rep.*, vol. 15, no. 4, pp. 997–1000, 2006.
- [9] V. Kodogiannis and J. N. Lygouras, "Neuro-fuzzy classification system for wireless-capsule endoscopic images," *Int. J. Elect. Comput. Syst. Eng.*, vol. 2, no. 1, pp. 55–63, 2008.
- [10] C. S. Lima, D. Barbosa, J. Ramos, A. Tavares, L. Monteiro, and L. Carvalho, "Classification of endoscopic capsule images by using color wavelet features, higher order statistics and radial basis functions," in *Proc. 30th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc.*, Aug. 2008, pp. 1242–1245.
- [11] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. Int. Conf. Learn. Represent.*, 2015, pp. 1–14.
- [12] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2016, pp. 770–778.
- [13] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, "Densely connected convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jul. 2017, pp. 4700–4708.
- [14] X. Li, H. Zhang, X. Zhang, H. Liu, and G. Xie, "Exploring transfer learning for gastrointestinal bleeding detection on small-size imbalanced endoscopy images," in *Proc. 39th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)*, Jul. 2017, pp. 1994–1997.
- [15] J. Yogapriya, V. Chandran, M. G. Sumithra, P. Anitha, P. Jenopaul, and C. Suresh Gnana Dhas, "Gastrointestinal tract disease classification from wireless endoscopy images using pretrained deep learning model," *Comput. Math. Methods Med.*, vol. 2021, pp. 1–12, Sep. 2021.
- [16] Y. Wang, Z. Feng, L. Song, X. Liu, and S. Liu, "Multiclassification of endoscopic colonoscopy images based on deep transfer learning," *Comp. Math. Methods Med.*, vol. 2021, Jul. 2021, Art. no. 2485934.
- [17] S. Shichijo et al., "Application of convolutional neural networks in the diagnosis of Helicobacter pylori infection based on endoscopic images," *EBioMedicine*, vol. 25, pp. 106–111, Nov. 2017.
- [18] S. V. Georgakopoulos, D. K. Iakovidis, M. Vasilakakis, V. P. Plagianakos, and A. Koulaouzidis, "Weakly-supervised convolutional learning for detection of inflammatory gastrointestinal lesions," in *Proc. IEEE Int. Conf. Imag. Syst. Techn. (IST)*, Oct. 2016, pp. 510–514.
- [19] T. Hirasawa, K. Aoyama, T. Tanimoto, S. Ishihara, and T. Tada, "Application of artificial intelligence using a convolutional neural network for detecting gastric cancer in endoscopic images," *Gastric Cancer*, vol. 21, no. 4, pp. 653–660, Jul. 2018.
- [20] D. K. Iakovidis, S. V. Georgakopoulos, M. Vasilakakis, A. Koulaouzidis, and V. P. Plagianakos, "Detecting and locating gastrointestinal anomalies using deep learning and iterative cluster unification," *IEEE Trans. Med. Imag.*, vol. 37, no. 10, pp. 2196–2210, Oct. 2018.
- [21] R. Zhang et al., "Automatic detection and classification of colorectal polyps by transferring low-level CNN features from nonmedical domain," *IEEE J. Biomed. Health Informat.*, vol. 21, no. 1, pp. 41–47, Jan. 2017.
- [22] K. Pogorelov et al., "Deep learning and hand-crafted feature based approaches for polyp detection in medical videos," in *Proc. IEEE 31st Int. Symp. Comput.-Based Med. Syst. (CBMS)*, Jun. 2018, pp. 381–386.
- [23] M. Billah and S. Waheed, "Gastrointestinal polyp detection in endoscopic images using an improved feature extraction method," *Biomed. Eng. Lett.*, vol. 8, no. 1, pp. 69–75, Feb. 2018.
- [24] M. A. Styner et al., "Cascaded deep decision networks for classification of endoscopic images," in *Proc. SPIE*, vol. 10133, 2017, pp. 642–656.
- [25] X. Zhang et al., "Gastric precancerous diseases classification using CNN with a concise model," *PLoS ONE*, vol. 12, no. 9, Sep. 2017, Art. no. e0185508.
- [26] S. Wang, Y. Yin, D. Wang, Z. Lv, Y. Wang, and Y. Jin, "An interpretable deep neural network for colorectal polyp diagnosis under colonoscopy," *Knowl.-Based Syst.*, vol. 234, Dec. 2021, Art. no. 107568.
- [27] A. Bochkovskiy, C.-Y. Wang, and H.-Y. Mark Liao, "YOLOv4: Optimal speed and accuracy of object detection," 2020, *arXiv:2004.10934*.

- [28] T. He, Z. Zhang, H. Zhang, Z. Zhang, J. Xie, and M. Li, "Bag of tricks for image classification with convolutional neural networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2019, pp. 558–567.
- [29] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent.*, 2015, pp. 234–241.
- [30] Y. Yuan et al., "Densely connected neural network with unbalanced discriminant and category sensitive constraints for polyp recognition," *IEEE Trans. Autom. Sci. Eng.*, vol. 17, no. 2, pp. 574–583, Apr. 2020.
- [31] X. Xing, Y. Yuan, and M. Q.-H. Meng, "Zoom in lesions for better diagnosis: Attention guided deformation network for WCE image classification," *IEEE Trans. Med. Imag.*, vol. 39, no. 12, pp. 4047–4059, Dec. 2020.
- [32] J.-S. Yu, J. Chen, Z. Q. Xiang, and Y.-X. Zou, "A hybrid convolutional neural networks with extreme learning machine for WCE image classification," in *Proc. IEEE Int. Conf. Robot. Biomimetics (ROBIO)*, Dec. 2015, pp. 1822–1827.
- [33] R. Zhu, R. Zhang, and D. Xue, "Lesion detection of endoscopy images based on convolutional neural network features," in *Proc. 8th Int. Congr. Image Signal Process. (CISP)*, Oct. 2015, pp. 372–376.
- [34] M. Billah, S. Waheed, and M. M. Rahman, "An automatic gastrointestinal polyp detection system in video endoscopy using fusion of color wavelet and convolutional neural network features," *Int. J. Biomed. Imag.*, vol. 2017, pp. 1–9, Aug. 2017.
- [35] X. Guo, L. Zhang, Y. Hao, L. Zhang, Z. Liu, and J. Liu, "Multiple abnormality classification in wireless capsule endoscopy images based on EfficientNet using attention mechanism," *Rev. Sci. Instrum.*, vol. 92, no. 9, Sep. 2021, Art. no. 094102.
- [36] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn.*, 2019, pp. 6105–6114.
- [37] J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, Jun. 2018, pp. 7132–7141.
- [38] Y. Zou, L. Li, Y. Wang, J. Yu, Y. Li, and W. J. Deng, "Classifying digestive organs in wireless capsule endoscopy images based on deep convolutional neural network," in *Proc. IEEE Int. Conf. Digit. Signal Process. (DSP)*, Jul. 2015, pp. 1274–1278.
- [39] H. Takiyama et al., "Automatic anatomical classification of esophagogastroduodenoscopy images using deep convolutional neural networks," *Sci. Rep.*, vol. 8, no. 1, p. 7497, May 2018.
- [40] C. Szegedy et al., "Going deeper with convolutions," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Jun. 2015, pp. 1–9.
- [41] A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. De Mathelin, and N. Padoy, "EndoNet: A deep architecture for recognition tasks on laparoscopic videos," *IEEE Trans. Med. Imag.*, vol. 36, no. 1, pp. 86–97, Jan. 2017.
- [42] S. Petscharnig, K. Schöffmann, and M. Lux, "An inception-like CNN architecture for GI disease and anatomical landmark classification," in *Proc. CEUR Workshop*, vol. 1984, 2017, pp. 1–3.
- [43] W. Wang, X. Yang, X. Li, and J. Tang, "Convolutional-capsule network for gastrointestinal endoscopy image classification," *Int. J. Intell. Syst.*, vol. 37, no. 9, pp. 5796–5815, Sep. 2022.
- [44] A. Vaswani et al., "Attention is all you need," in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 30, 2017, pp. 1–11.
- [45] D. Zhou et al., "DeepViT: Towards deeper vision transformer," 2021, *arXiv:2103.11886*.
- [46] W. Wang et al., "Pyramid vision transformer: A versatile backbone for dense prediction without convolutions," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Oct. 2021, pp. 568–578.
- [47] H. Lin, X. Cheng, X. Wu, and D. Shen, "CAT: Cross attention in vision transformer," in *Proc. IEEE Int. Conf. Multimedia Expo (ICME)*, Jul. 2022, pp. 1–6.
- [48] W. Wang et al., "CrossFormer: A versatile vision transformer hinging on cross-scale attention," in *Proc. Int. Conf. Learn. Represent.*, 2021, pp. 1–15.
- [49] C.-F. Chen, R. Panda, and Q. Fan, "RegionViT: Regional-to-local attention for vision transformers," in *Proc. Int. Conf. Learn. Represent.*, 2021, pp. 1–19.
- [50] H. Wu et al., "CvT: Introducing convolutions to vision transformers," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Oct. 2021, pp. 22–31.
- [51] Y. Li, K. Zhang, J. Cao, R. Timofte, and L. Van Gool, "LocalViT: Bringing locality to vision transformers," 2021, *arXiv:2104.05707*.

- [52] Z. Dai, H. Liu, Q. V. Le, and M. Tan, "CoAtNet: Marrying convolution and attention for all data sizes," in *Proc. Adv. Neural Inf. Process. Syst.*, vol. 34, 2021, pp. 3965–3977.
- [53] X. Wang, J. Liu, S. Chen, and G. Wei, "Effective light field de-occlusion network based on Swin transformer," *IEEE Trans. Circuits Syst. Video Technol.*, early access, Dec. 1, 2022, doi: [10.1109/TCSVT.2022.3226227.](http://dx.doi.org/10.1109/TCSVT.2022.3226227)
- [54] Y. Wang, Y. Qiu, P. Cheng, and J. Zhang, "Hybrid CNN-transformer features for visual place recognition," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 33, no. 3, pp. 1109–1122, Mar. 2023.
- [55] K. Pogorelov et al., "KVASIR: A multi-class image dataset for computer aided gastrointestinal disease detection," in *Proc. ACM Multimedia Syst. Conf.*, 2017, pp. 164–169.
- [56] H. Borgli et al., "*HyperKvasir*, a comprehensive multi-class image and video dataset for gastrointestinal endoscopy," *Sci. Data*, vol. 7, no. 1, p. 283, Aug. 2020.

![](assets/pictures/_page_9_Picture_7.jpeg)

Xin Yang (Member, IEEE) received the Ph.D. degree from the University of California at Santa Barbara, Santa Barbara, in 2013. She worked as a Post-Doctoral Researcher with the Learning-Based Multimedia Laboratory, UCSB (2013–2014). She is currently a Professor with the School of Electronic Information and Communications, Huazhong University of Science and Technology. She has published over 100 technical papers, including IEEE TRANSACTIONS ON PATTERN ANALY-SIS AND MACHINE INTELLIGENCE, *IJCV*, IEEE

TRANSACTIONS ON MEDICAL IMAGING, MedIA, CVPR, ECCV, and MM. She has coauthored two books and holds three U.S. patents. Her research interests include computer vision and medical image analysis. She has served as an Associate Editor for the IEEE TRANSACTIONS ON MEDICAL IMAGING and *Multimedia Systems*. She is a member of ACM.

![](assets/pictures/_page_9_Picture_10.jpeg)

Wei Wang received the B.E. degree from the Nanjing University of Science and Technology, Nanjing, China, in 2014, where he is currently pursuing the Ph.D. degree. From 2018 to 2019, he was an Intern with the National University of Singapore, Singapore. His research interests include computer aided medical image classification and segmentation.

![](assets/pictures/_page_9_Picture_12.jpeg)

Jinhui Tang (Senior Member, IEEE) received the B.E. and Ph.D. degrees from the University of Science and Technology of China, Hefei, China, in 2003 and 2008, respectively. He is currently a Professor with the Nanjing University of Science and Technology, Nanjing, China. He has authored more than 150 articles in top-tier journals and conferences. His research interests include multimedia analysis and computer vision. He is a fellow of IAPR. He was a recipient of best paper awards in ACM MM 2007 and ACM MM Asia 2020 and the Best Paper

Runner-Up in ACM MM 2015. He has served as an Associate Editor for the IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING and IEEE TRANSACTIONS ON MULTIMEDIA.