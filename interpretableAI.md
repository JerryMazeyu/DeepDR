# Interpretable AI
[TOC]

## 1 可解释性概述
深度学习在“端到端”模式下、通过标注大量数据来进行误差后向传播而优化参数的学习方法被比喻为一个“黑盒子”，解释性较弱。可解释性指算法要对特定任务给出清晰概括，并与人类世界中已定义的原则或原理联结。在诸如自动驾驶、医疗和金融决策等“高风险”领域，利用深度学习进行重大决策时，往往需要知晓算法所给出结果的依据。因此，透明化深度学习的“黑盒子”，使其具有可解释性，具有重要意义。
### 1.1 什么是可解释性？
对于这个问题，没有数学上的定义，相关的非数学定义有：   

* 可解释性是人们可以理解决策原因的程度。  
（Interpretability is the degree to which a human can understand the cause of a decision<sup>[1]</sup>.）  
* 可解释性是人类可以一致地预测模型结果的程度。  
（Interpretability is the degree to which a human can consistently predict the model’s result<sup>[2]</sup>.）

### 1.2 可解释性的重要性
有些模型可能不需要解释，因为它们是在低风险的环境中使用的，这意味着错误不会造成严重后果（例如，电影推荐系统），或者该方法已经得到了广泛的研究和评估（例如，光学字符识别）。  
对可解释性的需求源于问题形式化的不完全性<sup>[3]</sup>，这意味着对于某些问题或任务，仅仅获得预测是不够的。该模型还必须说明预测是如何进行的，因为正确的预测只能部分解决您的原始问题。  
这也就是说，对于**风险很高**的问题，例如医疗与自动驾驶等，对于可解释性的需求更大。  
除此之外，对于深度学习模型中，可解释性的重要性还在于：可以帮助我们突破几个关键的瓶颈，例如：从很少的标注中学习；从语义级别的人机交互中学习，以及在语义层面上调试模型的表征。

### 1.3 可解释性的分类
 **模型固有（intrinsic） or 事后生成（post hoc）？**  
这个标准区分了实现可解释性的机制：限制模型的复杂程度（模型固有，eg：决策树）；在训练后通过分析方法对模型解释（事后生成，eg：按层表述CNN的特征图）  
模型固有的可解释性是指由于其简单的结构而被认为是可解释的机器学习模型，例如短决策树或稀疏线性模型。  
而事后生成的解释性则是将复杂机器学习模型应用一系列可解释方法，从而达到可解释性，当然，那些已经可解释的模型也可以被应用到这些方法。其中解释的方法主要分为：   

* **特征摘要统计量（Feature summary statistic）**：对每个特征在模型中的重要性提出解释，例如XGBoost中的feature_importance，可以对模型考虑因素程度提供参考，有些模型可以提供更复杂的特征的程度指标，例如成对的要素交互强度，该结果由每个要素对的数字组成。
* **特征摘要可视化（Feature summary visualization）**：对于特征摘要统计量的可视化表示，有时只有在可视化的情况下特征摘要才是有意义的，例如偏相关图是显示特征和平均预测结果的曲线，呈现偏相关性的最佳方法是实际绘制曲线而不是打印坐标。
* **模型内部机理（例如学习到的权重）（Model internals）**：模型固有解释性可以被分类到这一类，有时这种解释机制和特征摘要统计量的界限也是模糊的，例如线性模型中系数即是模型内部的参数也是特征摘要的统计量。
* **典型数据（Data point）**：对预测数据返回一些与之类似的数据，这类模型会对数据进行一些变化，然后找到其相似的数据点，例如在SVM中对于数据的核变换。  
* **利用内部可解释模型逼近（Intrinsically interpretable model）**：利用可解释的模型对“黑箱”模型进行逼近，例如对于任意函数f(x)进行泰勒展开。 
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-16-%E5%8F%AF%E8%A7%A3%E9%87%8A-1.png)

综上所述，由于深度学习模型具有极高的模型复杂度，可解释性人工智能可以被理解为**利用解释方法应用在深度学习模型中，以达到深度学习模型的可解释性**。 

## 2 人工智能的解释方法
根据1.3节的论述，已将可解释性方法进行了介绍。在深度学习模型中，主要的方式为**特征摘要可视化**，**非可视化方法**以及**与模型无关的方法**，下面依次进行介绍。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-17-%E5%8F%AF%E8%A7%A3%E9%87%8A%E6%A8%A1%E5%9E%8B.png)
### 2.1 可解释性深度学习的可视化方法
可视化的深度学习大部分集中在卷积神经网络中，深度学习的可视化方法主要分为五个方向：<sup>[4]</sup> 

* **网络中间层特征图的可视化**：卷积神经网络中间层中的特征图的可视化是探索神经单元中视觉模式的最直接的可视化方式。首先，基于梯度的方法<sup>[5][6]</sup>仍然是主流趋势，主要的做法是计算给定CNN单元的输入梯度（特征图），通过反向传播，利用梯度来估计让达到让CNN单元得分最大（loss最小）的特征图。Olah开发了一个[工具箱](https://distill.pub/2017/feature-visualization)，可以用来显示在预先训练好的CNN在不同层之间的特征图<sup>[7]</sup>;其次，利用反卷积也是一种让特征图可视化的典型技术，虽然反卷积并不能保证其结果可以准确的反映图像表征，但是也可以用这种方式作为工具间接的表示特征图。Zhou提出了一种准确计算特征图中特征图分辨率可接受域的方法<sup>[8]</sup>，对接受域的准确估计有助于人们理解卷积核的表示。
* **CNN表征的分析**：有些方法超越了传统的对于特征图的可视化，有助于卷积神经网络的特征进行深度理解。这类研究大致有五个方向：  
首先是对于卷积神经网络的**全局分析**，Szegedy探究了每一个卷积神经单元的含义<sup>[9]</sup>，Yosinski分析了中间层的卷积核的可转移性<sup>[10]</sup>，Aubry分析了一个预训练的CNN分不同类别的时候的特征分布<sup>[11]</sup>；  
第二个方向是研究对分类产生影响的**主要特征图区域**，这和网络层中间的特征图可视化类似，Ribeiro提出LIME模型对模型输出结果敏感的重点区域进行可视化提取<sup>[12]</sup>；  
估计特征空间中的**脆弱点**也是对CNNs表征分析的流行方向，Su等人开发出计算CNN的对抗样本，这些研究的目标旨在估计输入图像产生最小噪声的情况下，可以改变最终的预测结果<sup>[13]</sup>，Koh提出了一个计算对抗样本的影响函数<sup>[14]</sup>，该函数可以提供合理的方法来创建训练样本来攻击CNN，修复训练集并进一步调试CNN；  
第四个研究方向是**基于对网络特征空间的分析来完善网络表示**，Lakkaraju提出了一种在弱监督下预训练CNNs上发现未知模式的方法<sup>[15]</sup>，此方法将整个样本空间分为数千个伪类别，并假设有一个分类完美的CNN可以通过对这些伪类别对样本进行识别，这些类别对应了特定的真实类别。为了对一个情绪分类的未训练的网络进行训练，Hu采用了一个预训练的网络作为“老师”对神经网络进行监督，并且构建
蒸馏损失用以监督蒸馏神经网络，以获得更有意义的网络表示形式<sup>[16]</sup>。  
最后，Zhang提出了一种方法用于发现**CNN的潜在偏差**<sup>[17]</sup>，CNN的偏差指即使在测试集上表现好的网络也不一定真正的能够说明模型的正确性，有时学到的特征其实并不是一个真正的正确特征，而是一个无关特征，例如下图所示，当遮住嘴巴后反而准确率提升，而涂上口红后反而下降。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-16-160314.png)
* **可解释的R-CNN**：  
在医疗图像和自动驾驶的区域，图像分割问题是一个需要解释的问题。Wu<sup>[18]</sup>提出了一种AOG解析算子（AOG parsing operator）来取代R-CNN中常用的RoI池化算子（RoI Pooling Operator），因此其提出的方法可以适用于很多基于卷积神经网络的顶尖目标检测系统。AOG解析算子的目标是同时利用自顶向下的分层综合语法模型的严密可解释性，以及以自底向上的端到端形式训练得到的深度神经网络的判别能力。在检测中，将使用从AOG中实时得到的最佳解析树（parse tree）来解释边界盒（bounding box），这个边界盒可以看做为解释检测而生成的提取基本原理。
* **胶囊网络**<sup>[19]</sup>：  
胶囊网络是一种新的用来代替神经元的单元，胶囊表示图像中特定实体的各种特征。一种非常特殊的特征是图像中实例化实体的存在。实例化实体是诸如位置、大小、方向、变形、速度、反照率、色调、纹理等参数。表示其存在的一个简单方法是使用单独的逻辑单元，其输出是实体存在的概率。为了得到比CNN更好的结果，我们应该使用一个迭代的协议路由机制。这些特性称为实例化参数。在经典的CNN模型中是没有得到图像中目标的这些属性的。平均/最大池层减少了一组信息的大小，同时减少了其尺寸大小。

### 2.2 可解释性深度学习非可视化方法 
* **disentangled representations learning（解耦表示学习）**<sup>[20]</sup>：  
解耦表示学习大致的意思是每一个表征可以对应到表征向量的不同维度上去。一个disentangled representation未必非要是构建在neuron之上，可以是任意形式的数据以及依赖于这个数据的问题抽象方式，例如可以是一个有向图，disentangled representation表示的是图中的边，而问题本身需要在这个有向图上寻找最短路径。  
这个问题在国际机器学习顶会ICML公布2019年最佳论文中Francesco Locatello做出了研究，无监督学习解耦表示背后的关键思想是，真实世界数据是由一些变量的解释因子生成的，这些因子可以通过无监督学习算法恢复，最先进的无监督耦合学习很大程度上基于变分自编码器（VAEs）。更多的参考详见[这里](https://github.com/sootlasten/disentangled-representation-papers)。  
* **利用评价矩阵评价可解释性**：  
对于神经网络的可解释性现在的研究比较少，但是这是让神经网络的有效表示能得到发展的一个很有价值的领域，相关的研究主要是Bau的卷积核可解释性指标<sup>[21]</sup>（filter interpretability）和Zhang的区域不稳定性<sup>[22]</sup>（location instability）。  
Bau为卷积核定义了六种语义：物体，零件，场景，纹理，材料和颜色。这个指标提供了**一个激活的神经元的图像分辨率感受野和像素级别的语义标注之间的匹配度**。例如，某一个神经元的激活程度往往和某一个特定的语义相关（如颜色），那么则可以认为这个神经元代表了这个特定的语义。具体的计算参考原文，最终可以得到一个Pf,k，其中f代表某个层，而k代表六种语义之一，Pf,k代表某个层对具体语义的相关性。  
Zhang提出了区域不稳定性去评估**一个神经元与物体某一个部分的表示匹配度**。在文中，假设某一个层可以检测出有关某一类物体的某一些特征点，而有关于某一类物体的一些固定点之间的相对位置是固定的，例如下图可以看出三个蓝色的点之间的相对位置是同样的，那么需要利用最短路算法找到一个其中的点，并计算一组测试图像之间的这些点的偏差，这个偏差可以作为一个神经层的区域不稳定性。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-17-030813.png)  
* **决策树逼近**：  
在卷积神经网络中，我们认为高层的神经层主要识别复合特征，例如某一层鸟类分类网络的神经元识别脚和头，如下图所示：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-17-042949.png)
那么，Zhang提出利用解释图算法（例如决策树、GNN等）将这些复合特征解耦，将图像自然的展示出来<sup>[23]</sup>。例如下图：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-17-043421.png)


## 3 针对医疗问题的可解释性研究
### 3.1 GAP代替FC
传统分类的深度学习模型中多采用全连接层对模型最终结果进行分类，这种分类方式缺乏可解释性并且参数量大，使用全局平均池化代替，直接将最后输出的特征图作为指标输出到模型，比如有10个类，就在最后输出10个特征图，每个特征图中的值加起来求平均值，这十个数字就是对应的概率或者叫置信度。然后把得到的这些平均值直接作为属于某个类别的置信指标，再输入softmax中分类，更重要的是实验效果并不比用FC差。
### 3.2 CAM方法
CAM（Class Activation Mapping）同热力成像图类似，是一种利用特征图激活情况理解输出层神经元的可视化手段。在卷积神经网络结构中，特征图是通过不断卷积压缩特征，从局部到整体，最后一层卷积层的提取特征信息最为丰富，CAM可以利用这层信息，并通过GAP连接的方式为这些特征图赋予相应的权重，表示各特征图的重要程度，依据权重对这些特征图做加权累和，将累加后的矩阵上采样至原图大小，使用热成像颜色板进行着色，可视化出最终判别器关注的区域，用以理解判别器的判别过程。这种方式正是2.1节中网络中间层特征图的可视化的应用。
### 3.3 基于语义的特征描述
以一个检测肿瘤的分类网络为例：一个能够在热度图像中重点关注肿瘤区域，弱视其他区域的模型有理由让我们相信论文训练的模型在利用对的信息做对的事情。但是可视化的方法只为我们提供了在图像级别上的模糊描述，告诉我们模型重点分析了肿瘤区域的特征，但是没有告诉我们分析了那些特征，换一种说法就是缺乏语义上的描述。 
针对这个问题，付贵山<sup>[24]</sup>提出了这样一种方法：利用现有的医学指标（例如乳腺影像学报告及数据系统，该系统针对乳腺鲴靶摄影影像进行了规范的、详尽的描述，并对乳腺疾病分级提供了一系列科学标准）中可解释特征和深度学习深层抽象特征的回归模型，回归模型能够将深度学习不可解释的抽象特征映射到可解释的语义特征空间，在可解释的语义特征空间中，我们可以观测数值上的偏差和各特征的重要程度。 在文中，对于不可解释的抽象特征选用基于灰度共生矩阵提出了综合考虑像素灰度和梯度的**灰度梯度共生矩阵**。对于这样的问题，**由于将无监督问题转换成了有监督的回归问题，就同时有了评价的指标（R<sup>2</sup>，MSE等）**。  
### 3.4 最新的应用于医疗的方法：beta-VAE。
有一个方法可以帮助我们实现解耦表示，也就是让嵌入中的每个元素对应一个单独的影响因素，并能够将该嵌入用于分类、生成和零样本学习。该算法是由 DeepMind 实验室基于变分自编码器开发的，相比于重构损失函数（restoration loss），该算法更加注重潜在分布与先验分布之间的相对熵。
有人利用这种模型应用于[心电图数据进行了分析](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/89899601)，可以发现，第5个特征对心跳形式的影响很大，第8个代表了心脏病的情况（蓝色心电图代表有梗塞症状，而红色则相反），第10个特征可以轻微地影响脉博。

## 4 如何评价可解释性？
上述提到的对于模型的可解释处理中有说到关于检验可解释性优劣的一些量化方法与评价指标，本章对其专门列出一章进行扩展解释。
### 4.1 基于特征（feature-based）的方法
结合第二节中所讲到的语义表示结果（例如2.1节中指出的对于主要特征区域的划分(CAM)方法），经过人为总结、并将其抽象成指定特征，然后对测试集进行特征消融，除去该特征，观察结果的改变并作回归查看结果：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-17-064600.png)
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-17-064724.png)
### 4.2 基于人工的方法
将说明放入产品中，并由最终用户进行测试。想象一下带有机器学习组件的骨折检测软件，该软件可以定位和标记X射线中的裂缝。在应用程序级别，放射科医生会直接测试骨折检测软件以评估模型。这需要良好的实验设置，并了解如何评估质量，一个良好的基准始终是人类在解释同一决定方面的能力。  
https://arxiv.org/abs/1811.11839
### 4.3 针对某个神经层  
这主要参考2.2节中**利用评价矩阵评价可解释性**。  
### 






## 参考文献
[1][Miller and Tim. “Explanation in artificial intelligence: Insights from the social sciences.” arXiv Preprint arXiv:1706.07269. (2017).](https://arxiv.org/abs/1706.07269)  
[2][Kim, Been, Rajiv Khanna, and Oluwasanmi O. Koyejo. “Examples are not enough, learn to criticize! Criticism for interpretability.” Advances in Neural Information Processing Systems. (2016).](http://papers.nips.cc/paper/6300-examples-are-not-enough-learn-to-criticize-criticism-for-interpretability)  
[3][Doshi-Velez, Finale and Been Kim. “Towards a rigorous science of interpretable machine learning,” no. Ml: 1–13. (2017).](http://arxiv.org/abs/1702.08608)  
[4][Quanshi Zhang and Song-Chun Zhu. “Visual Interpretability for Deep Learning: a Survey” arXiv:1802.00614. (2018).](https://arxiv.org/abs/1802.00614)  
[5][Matthew D. Zeiler, Rob Fergus. Visualizing and understanding convolutional networks. ECCV. (2014)](https://arxiv.org/abs/1311.2901)  
[6][Aravindh Mahendran and Andrea Vedaldi. Understanding deep image representations by inverting them.CVPR. (2015).](https://arxiv.org/abs/1412.0035)  
[7][Chris Olah, Alexander Mordvintsev and Ludwig Schubert. Feature visualization. (2017).](https://distill.pub/2017/feature-visualization)  
[8][Bolei Zhou, Aditya Khosla, Agata
Lapedriza, Aude Oliva, and Antonio Torralba. Object detectors emerge in deep scene cnns. ICRL. (2015).](https://arxiv.org/abs/1412.6856)  
[9][Christian Szegedy, Wojciech
Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian
Goodfellow, and Rob Fergus. Intriguing properties of
neural networks. arXiv:1312.6199. (2014).](https://arxiv.org/abs/1312.6199)  
[10][Jason Yosinski, Jeff Clune, Yoshua
Bengio, and Hod Lipson. How transferable are features
in deep neural networks? NIPS. (2014).](http://xueshu.baidu.com/usercenter/paper/show?paperid=ec066d96a39872b895b90eed058ff52b&site=xueshu_se)  
[11][M. Aubry, D. Maturana, A. Efros,
B. Russell, and J. Sivic. Seeing 3d chairs: Exemplar part-
based 2d-3d alignment using a large dataset of cad models.
CVPR. (2014).](https://www.computer.org/10.1109/CVPR.2014.487)  
[12][Marco Tulio Ribeiro, Sameer Singh,
and Carlos Guestrin. “why should i trust you?” explaining
the predictions of any classifier. KDD. (2016).](https://arxiv.org/abs/1602.04938)  
[13][Jiawei Su, Danilo Vasconcellos Vargas, and Sakurai Kouichi. One pixel attack for fooling deep neural networks.arXiv:1710.08864. (2017).](https://arxiv.org/abs/1710.08864)  
[14][PangWei Koh and Percy Liang. Understanding black-box predictions via influence functions.ICML. (2017).](http://xueshu.baidu.com/usercenter/paper/show?paperid=a979ae445067971000e3958473b37bdd&site=xueshu_se)  
[15][Himabindu Lakkaraju, Ece Kamar, Rich Caruana, and Eric Horvitz. Identifying unknown unknowns in the open world: Representations and policies
for guided exploration. AAAI. (2017).](https://arxiv.org/abs/1610.09064v3)  
[16][Zhiting Hu, Xuezhe Ma, Zhengzhong Liu, Eduard Hovy, and Eric P. Xing. Harnessing deep neural networks with logic rules. In ACL. (2016).](https://arxiv.org/abs/1603.06318v3)  
[17][Quanshi Zhang, Ruiming Cao, Ying Nian Wu, and Song-Chun Zhu. Mining object parts from cnns via active question-answering. CVPR. (2017).](https://arxiv.org/abs/1704.03173)  
[18][Tianfu Wu, Wei Sun, Xilai Li, Xi Song and Bo Li. Towards Interpretable R-CNN by Unfolding Latent Structures. 	arXiv:1711.05226. (2018)](https://arxiv.org/abs/1711.05226)  
[19][Sara Sabour, Nicholas Frosst, and Geoffrey E. Hinton. Dynamic routing between capsules. In NIPS. (2017).](https://arxiv.org/abs/1710.09829)  
[20][Rui Shu, Shengjia Zhao and Mykel J. Kochenderfer. RETHINKING STYLE AND CONTENT DISENTANGLEMENT IN VARIATIONAL AUTOENCODERS. ICLR. (2018)](https://openreview.net/pdf?id=B1rQtwJDG)  
[21][David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. CVPR. (2017).](https://arxiv.org/abs/1704.05796)  
[22][Q. Zhang, R. Cao, F. Shi, Y.N. Wu, and S.-C. Zhu. Interpreting cnn knowledge via an explanatory graph. AAAI. (2018).](https://arxiv.org/abs/1708.01785)  
[23][Quanshi Zhang, Yu Yang, Ying Nian Wu, and Song-Chun Zhu. Interpreting cnns via decision trees. arXiv:1802.00121. (2018).](https://arxiv.org/abs/1802.00121)  
[24][付贵山. 深度学习乳腺超声图像分类器及其可解释性研究[D].哈尔滨工业大学. (2019).](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202001&filename=1019648871.nh&uid=WEEvREcwSlJHSldRa1FhdXNzY2Z1UmViY0lTRE11N2M4UFpJcTVvMDlqZz0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!&v=MTg2NjM2RjdXOEZ0bkxycEViUElSOGVYMUx1eFlTN0RoMVQzcVRyV00xRnJDVVI3cWZaT2RwRnl2blZiN0xWRjI=)
[25][Yoshua Bengio†, Aaron Courville, Pascal Vincent. Representation Learning: A Review and New Perspectives. arXiv:1206.5538v3 [cs.LG]. (2014)](https://arxiv.org/pdf/1206.5538.pdf)


