# DeepDR Report
竞赛旨在根据**眼底图像**完成糖尿病视网膜病变的**病症评级**、**图像质量评估**以及**构造可迁移学习模型**三项任务。下面针对不同的问题，作出对于该问题的报告。

## Challenge1：病症分级
### 问题分析
本题旨在根据高清的常规眼底图像对糖尿病性视网膜病变进行分级，该病变被分成五级，编号为0、1、2、3、4，分级标准为：

编号 | 名称 | 特征
:-: | :-: | :-: |
0 | 无明显病变 | 无明显异常迹象
1 | 轻度NPDR | 仅存在微动脉瘤
2 | 中度NPDR | 不仅是微动脉瘤，其程度在轻度和重度之间
3 | 重度NPDR | 中度NPDR症状外还有其他严重症状（出血等）
4 | PDR | 重度NPDR外还有新血管生成或玻璃体/视网膜前出血

* 可以看出这是一个图像多分类的问题，从[图像实例](https://isbi.deepdr.org/data.html)以及相关文献中可以看出，常规眼底图像的**明暗分布情况**（正常图像透明，几乎没有暗斑，而其他图像有明显黑斑）、**图像整体颜色**（中度及以上病变图像明显泛红）、**血管清晰程度**（重度及以上几乎看不到细血管）以及**是否有瘤**是分级的关键图像信息。
* 与其他类似图像竞赛不同的是，本次竞赛中每只眼睛提供两幅图像：一幅以视盘为中心，另一幅以中央凹为中心。这可能涉及到一些多张图像融合以及特征提取的问题，或者是分别对两张图像进行建模再利用集成算法进行结果的评估。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-11-170136.png)

### 文献综述
在糖尿病性视网膜病变的研究中，简单的分级评价主要的做法是根据卷积神经网络去进行直接分类，Takahashi H<sup>[1]</sup>利用改进后的GoogLeNet进行分级，最终训练出的结果平均准确率为96％，值得注意的是，作者提到在疾病分级的任务AlexNet效果不佳，并且ResNet消耗内存很大，导致无法进行训练；Aiki Sakaguchi<sup>[2]</sup>利用图神经网络GNN进行，而且在特征工程方面，利用了提取关注区域（Region-Of-Interest）进行图像特征提取，重点捕获病灶区域进行，最终得到结果在挑战赛中比常规方法准确率高2.9%；Carson Lam<sup>[3]</sup>开发出一种自动视网膜病变的分级系统，同样是利用AlexNet和GoogLeNet作为卷积神经网络进行尝试，但是作者提出了一种利用受限的自适应直方图均衡滤波算法（CLAHE）进行特征提取，这种方式被证明可以将三元分类器的敏感度从0增加到29.4％，作者对于不同的优化策略和学习率等超参数进行了尝试，最终得到结论在GoogLeNet卷积神经网络上，学习率为1e-4、Adam优化算法、采用dropout策略的情况下达到了最优效果，测试集上达到了74.5%的准确率，值得注意的是，作者提出在四分类的情况下已经出现了欠拟合的情况，而他的数据量和本竞赛的大致相当，这值得注意；Porwal Prasanna<sup>[4]</sup>对2018年同样的挑战赛比赛情况作出了总结，提到在这个挑战中，参赛者们还是采用迁移学习的策略微调成熟的卷积神经网络的变体，参赛者们主要选择DenseNet网络或者深层聚合（DLA），并且提到图像的分辨率对于最终判别结果有着重要的影响；Li<sup>[5]</sup>开源了一个网络[CANET](https://github.com/xmengli999/CANet)，但并没有公布训练效果与readme；乔志强<sup>[6]</sup>提出了一种双塔结构的特征提取方式，利用采用经典的Inception-ResNet-v2网络和Xception网络进行双向分类，特别的，作者构造的模型接受的不是原始图像，而是进行U-Net分割后的特征图；王煜杰<sup>[7]</sup>针对色彩通道，提出RGB中绿色通道血管和背景差异明显，对提取血管和硬性渗出物质提取效果好，而对视盘的提取则是红色通道最佳；刘磊<sup>[8]</sup>主要针对UNet血管分割方法进行改进，提出了一种beta-dropout正则化代替UNet模型中的dropout策略，并且借鉴了注意力机制，进行了通道级别（channnel wise）的加权，并修改了损失函数为多级交叉熵。
### 解决方案
根据以往的论文以及思路，我的想法是针对数据特征提取（预处理与血管分割）、传统多分类任务以及模型融合分别提出解决方案，最终的成果将是对这三个问题的组合。

* **针对数据特征提取：**  
 首先，尝试将图像直接压缩成224像素，如果效果不佳则需要将一个图像分割成n小块进行处理，这样喂入模型的每一张原始图像将会是一个`n*224*224*3`的张量。对于图像，可以参考Carson Lam<sup>[1]</sup>的思路，进行Otsu方法和对比度受限的自适应直方图均衡化。处理后的效果为：

 ![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-09-171406.jpg)
 
 可以看出这样有效的进行了特征增强。  
 另外，可以根据外部的[眼底图像分割数据集](https://drive.grand-challenge.org/)进行血管的分割，这样可以将血管图作为新的特征喂入模型中，分割模型可以采用常见的[UNet](https://github.com/milesial/Pytorch-UNet)或者最新的[Detectron2](https://github.com/facebookresearch/detectron2)进行分割。

* **针对分类模型构建：**  
 现在主流的实现方式依然是要么直接利用CNN的现有变体，要么将它们组合起来以将输入图像划分类别。受欢迎的网络有AlexNet、GoogLeNet、ResNet、DenseNet或者深层聚合（DLA）结构。根据现实情况，采用预训练的DenseNet进行分类，或者加入注意力机制对图像进行不同权重的处理。  
 还有就是分类器的选择问题，既可以选择传统的SoftMax层进行多分类，也可以采用SVM这种非线型分类器对深度学习模型的特征输出做二次建模，得到分类结果，不过这需要尝试才能知道结果。

* **针对模型融合**   
由于本竞赛对于每一只眼睛会对应两幅图片（视盘中心、黄斑中心），那么就需要针对不同的图像类型分别建模，最后采用模型融合的方式输出最终结果。  
模型融合的方式主要是主要是**voting**，**averaging**，**bagging**，**boosting**，**blending**，以及**stacking**，其中bagging和boosting是集成学习的方法，基分类器主要是弱分类器（例如决策树），所以并不适合本模型。鉴于stacking是学习性的模型融合方法，现在被最常用于数据竞赛，可能会倾向与采用[blending或stacking](https://www.cnblogs.com/libin47/p/11169994.html)方法将模型进行融合。


### 评价指标与提交方法
* **评价指标**：[二次加权kappa系数](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview/evaluation)
Kappa系数是一种比例，代表着分类与完全随机的分类产生错误减少的比例。  
kappa=1 两次判断完全一致  
kappa>=0.75 比较满意的一致程度  
kappa<0.4 不够理想的一致程度  
* **提交方法**：上传一个包含数据、代码与结果的docker，其中结果的格式已经被指定、接口需要在项目中的根目录下。具体的命令行在竞赛的[说明](https://drive.google.com/file/d/18B7GVE_KE9COcS8KnwSZzfoaxoWjmRHB/view)中。


### 参考文献
[1] [Takahashi H, Tampo H, Arai Y, Inoue Y, Kawashima H. Applying artificial intelligence to disease staging: Deep learning for improved staging of diabetic retinopathy. PLoS One. 2017;12(6):e0179790. Published 2017 Jun 22. doi:10.1371/journal.pone.0179790](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5480986/)  
[2] [Aiki Sakaguchi, et al. Fundus Image Classification for Diabetic Retinopathy Using Disease Severity Grading](https://dl.acm.org/doi/abs/10.1145/3326172.3326198)  
[3] [Carson Lam，et al.Automated Detection of Diabetic Retinopathy using Deep Learning[J].AMIA Jt Summits Transl Sci Proc. 2018; 2018: 147–155.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/)  
[4] [Porwal Prasanna,Pachade Samiksha,Kokare Manesh,Deshmukh Girish,Son Jaemin,Bae Woong,Liu Lihong,Wang Jianzong,Liu Xinhui,Gao Liangxin,Wu TianBo,Xiao Jing,Wang Fengyan,Yin Baocai,Wang Yunzhi,Danala Gopichandh,He Linsheng,Choi Yoon Ho,Lee Yeong Chan,Jung Sang-Hyuk,Li Zhongyu,Sui Xiaodan,Wu Junyan,Li Xiaolong,Zhou Ting,Toth Janos,Baran Agnes,Kori Avinash,Chennamsetty Sai Saketh,Safwan Mohammed,Alex Varghese,Lyu Xingzheng,Cheng Li,Chu Qinhao,Li Pengcheng,Ji Xin,Zhang Sanyuan. IDRiD: Diabetic Retinopathy - Segmentation and Grading Challenge.[J]. Medical image analysis,2020,59.](https://www.sciencedirect.com/science/article/pii/S1361841519301033)  
[5] [X. Li, X. Hu, L. Yu, L. Zhu, C. Fu and P. Heng, "CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading," in IEEE Transactions on Medical Imaging.](https://github.com/xmengli999/CANet)  
[6] [乔志强. 基于深度学习的糖尿病眼底图像自动分类技术研究[D].北京邮电大学,2019.](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201902&filename=1019042221.nh&v=MTk4ODBlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSN3FmWk9kdEZ5M2hWTHJJVkYyNkY3TzhITlBPcnBFYlBJUjg=)  
[7] [王煜杰. 眼底图像病变区域的提取与识别[D].电子科技大学,2019.](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD202001&filename=1019850875.nh&v=MDcwMjhIdG5McXBFYlBJUjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSN3FmWk9kdEZ5em1VYnZKVkYyNkY3dTk=)  
[8] [刘磊. 基于深度神经网络的视网膜病变检测方法研究[D].中国科学技术大学,2019.](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFDLAST2019&filename=1019057814.nh&v=MTA3MDNMdXhZUzdEaDFUM3FUcldNMUZyQ1VSN3FmWk9kdEZ5em1WN3JPVkYyNkY3TzlHZG5OcTVFYlBJUjhlWDE=)

## Challenge2:图像质量评估
### 问题分析
本题的挑战是对眼底图像的质量进行评估，其中分成四个子任务，分别是**人为因素**、**清晰度**、**捕捉的范围**和**总体标准**，要求挑战者完成其中一个子任务即可。
其中各种衡量指标的[分类标准](https://isbi.deepdr.org/challenge2.html)如下：  

| 类型   | 评价标准                     | 级别 |
| -------- | -------------------------------- | ---- |
| 总体标准 | 可治愈                        | 1    |
|          | 不可治愈                     | 0    |
| 人为因素 | 不含人为因素               | 0    |
|          | 主动脉外侧部分，但区域小于1/4 | 1    |
|          | 不影响1/4的黄斑区         | 4    |
|          | 区域>1/4，<1/2                | 6    |
|          | 区域>1/2，但不完全覆盖后极 | 8    |
|          | 覆盖整个后极               | 10   |
| 清晰度 | 可识别一级血管弓         | 1    |
|          | 可识别二级血管弓，并识别少量病变 | 4    |
|          | 可识别三级血管弓，并识别一些病变 | 6    |
|          | 可识别三级血管弓，识别大部分病变 | 8    |
|          | 可识别三级血管弓，可识别所有病变 | 10   |
可以看出，这也是一个多分类的问题。样图如下：
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-12-123133.png)
对于四个子任务，关注的图像特点各有侧重：  

* 对于总体标准而言，需要提取图像的总体特征，是做一个二分类问题，主要的方式还是如任务1一样，利用**卷积神经网络做分类问题**，只是在最后的类别问题上从任务1的五分类问题变成了2分类问题，对于这样的问题，可以利用SVM做分类器进行分类，个人感觉这也可能是四个子任务中最容易的；
* 对于人为因素而言，可以从上图看出图（a）的整体呈像区域是与正常的图像不符，这说明对于这类图像关键点在于检测其整体呈像区域，这可能需要用到**边缘检测**的算法进行；
* 对于清晰度而言，根据其评价标准可以看出这正是需要对**血管进行检测**，在上一个问题中就提出过一些方法；
* 对于捕捉范围而言，主要是检测其视盘和黄斑是否在图像中，这可以用**目标检测**的算法去进行，例如YOLO或SSD等，但是如何去标注关键区域让机器去学习到黄斑和视盘的模式，这是一个值得思考的问题。

综上所述，本人主要针对子任务三：**清晰度质量评价**进行研究，这样选择的目的是因为：血管检测历来就是视网膜图像研究的一个重点，具有很强的现实价值；而且将血管形状提取出来的任务和问题一相辅相呈可以互相促进。
### 文献综述
文献综述分别针对整体的图像评价问题、对于眼底图像的质量评价问题与血管分割问题进行说明。  

* 首先，对于**整体的图像评价（IQA）问题**，主要是利用统计方法进行，其中最早提出的指标有Pearson相关系数、Spearman相关系数、超界比例、均方误差等系数，例如基于多元线性回归的主观质量评估方法（MOS）<sup>[1]</sup>, 或者利用EM算法将图像质量指标作为参数进行估计，这样做的合理性在于EM算法主要对含有隐变量的概率分布进行参数估计，而在图像中，我们无法预设像素值服从哪一种分布情况<sup>[2]</sup>。而这些方法并不适用于本任务，因为这些方法本质上是基于统计方法作出的无监督情况下的图像质量评价，因为在本任务中显然是一个监督问题，在数据中已经给出了关于眼底图像各类等级的标签。   
* 对于**眼底质量图像质量评价**方面，图像的质量评估主要应用在指纹质量识别和眼底质量识别中，所采用的方法依然主要基于卷积神经网络来进行，王佳阳<sup>[3]</sup>提出了基于深度残差网络（ResNet）与SVR混合模型进行眼底质量评估的算法，采用的方法主要是利用ResNet进行特征提取，再将特征传入SVR支持向量回归机中，这种方法其实并无新意，其次作者还提出了一种基于排序学习的质量评估方法，构建了ROFE网络图，将成对的两张图像分别传入ResNet中，最终结果做差作为新的特征传入网络中；闫晓葳<sup>[4]</sup>则利用随机森林方法进行质量评估，虽然其选用的分类器已经稍显落后，但是作者在文中提出了一种新的特征提取的方式：“首先将图像分块,通过curvelet变换进行预处理以减少图像噪声从而更好的选择图像中的各向异性块；然后,在经过预处理的图像中计算每个块的各向异性值,取阈值得图像中的各向异性块。”，这种方式是对于图像分块之后的二次筛选，利用这样被证明可以更好的进行特征处理，可以屏蔽部分负特征；王翠翠<sup>[5]</sup>利用三种机器学习分类器对其进行分类，从现在的角度看几乎没有做出任何的创新。  

可以看出在图像评价方向上大多数的研究方向都是集中在无监督情况下，利用统计学的方式进行。这是不难理解的，如果对于图像已经有了标签，假设这个指标是由专业医生根据经验打出的，那么图像的评价问题自然就转换成了利用深度学习模型去学习医生的行为，这样的行为变成了传统深度学习中多分类的问题了。于是，接下去将会对**血管分割任务**进行阐述：   

* 对于**血管分割任务**，现在最主流的方式是利用UNet进行分割，任务一中提到了王磊曾经利用改进的dropout策略优化了UNet；Jeamin Son<sup>[6]</sup>修改了UNet中全连接层的维度，使上采样层得到相同的输出维度，其次他们还修改了maxpooling层，目的是让病灶部位恰好放在图片被关注区域，作者还使用inverse pixel shuffling的方式对图片进行压缩，采用的是二元交叉熵作为loss；这位教授是专门研究眼底图像的，他的研究对本任务有较大启发；他在最近的研究中利用对抗生成网络（GAN）对眼底血管进行了精确分割<sup>[7]</sup>，本人认为这是一个十分有价值的研究，提供了血管分割任务中一个新的方向；Fengyan Wang<sup>[8]</sup>提出了一种新颖的基于CNN的基于U-Net的视网膜病变分割方法作为基本模型：它由三个阶段组成，第一阶段是用于获得初始分割蒙版的粗略分割模型，第二阶段是为进行假阳性归约而设计的级联分类器，最后，使用精细分割模型从以前的阶段得到结果；Lihong<sup>[9]</sup>开发了一种新颖的基于补丁的CNN模型，其中他们创新地将DenseNets与U-Net结合在一起以捕获更多上下文信息和多尺度特征；Li通过嵌入DLA结构开发了一种基于FCN的方法，由于病变位于分散且不规则的位置，DLA结构与FCN的嵌入可以更好地聚集来自局部和全局级别的语义和空间信息，从而增强了对病变的认识。


### 解决方案
主要对于**血管分割**任务是解决方案，通过阅读文献和以往竞赛的经验，我认为有如下一些思路去完成这样的任务：

* 通过**改进UNet**进行分割，这是一种现在看上去最主流的做法，UNet也类似于Encoder-Decoder架构，先做特征提取和压缩，再对其不断进行上采样。那么根据以往类似竞赛情况来看，我们完全可以在UNet结构中去调整，例如在UNet结构的“Encoder”部分部采用原本类似VGG的结构，而采用DenseNet这种更加厚的特征提取器，以便让特征可以更好的保留；其次，可以采用类似Fengyan Wang的模式，对模型进行串行式叠加，在Baseline：UNet的输出结果后进行分析，将其结果再喂入另一个模型中，这个模型可以是FCN；最后，我们可以参考刘磊的研究，复现其beta-dropout层，将其加入UNet中。
* 通过对抗生成式网络进行分割，这是来自Jeamin Son的研究结果，这是一种非常好的思路，并且作者开源了他的代码，完全可以在这个基础上对其进行改进。
* 通过FaceBook最新研究的detectron2分割模型进行分割，Detectron2是Facebook AI Research推出的最强目标检测平台，它实现了最新的目标检测算法，是对先前版本Detectron的完全重写，这是一个很好的结果，也容易复现。  

特别的，所有的关于血管分割的问题都需要利用到外部数据集，不过在任务一中我们可以看到是有这样的血管分割的数据集存在的。  
我的具体实现步骤可能会是首先利用原始UNet作为baseline，其次尝试运行detectron2平台进行分割，这是因为这个项目被维护的很好，应用应该比较便捷；如果效果不理想则考虑是否对UNet进行架构上的改进，我认为从头实现一个GAN并训练的成本太高，不会将其首先考虑。

### 参考文献
[1][W. Xuehui, Li Junli and C. Gang, "An Image Quality Estimation Model Based on HVS," TENCON 2006 - 2006 IEEE Region 10 Conference, Hong Kong, 2006, pp. 1-4.](https://ieeexplore.ieee.org/abstract/document/4142193)  
[2][L. Parra and H. H. Barrett, "List-mode likelihood: EM algorithm and image quality estimation demonstrated on 2-D PET," in IEEE Transactions on Medical Imaging, vol. 17, no. 2, pp. 228-235, April 1998.](https://ieeexplore.ieee.org/abstract/document/700734)  
[3][王佳阳. 眼部光学相干断层扫描图像的质量评价算法研究[D].西北大学,2019.](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201902&filename=1019663777.nh&v=MTM2OTViUElSOGVYMUx1eFlTN0RoMVQzcVRyV00xRnJDVVI3cWZaT2R0RkNya1VML09WRjI2RjdXK0hkYkxxSkU=)  
[4][闫晓葳. 基于随机森林的视网膜图像质量评价[D].山东大学,2015.](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201601&filename=1015372613.nh&v=MjA1ODU3Qy9ITmZOckpFYlBJUjhlWDFMdXhZUzdEaDFUM3FUcldNMUZyQ1VSN3FmWk9kdEZDcmtVTHZPVkYyNkc=)  
[5][王翠翠. 彩色眼底图像质量自动评估[D].东北大学,2015.](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CMFD&dbname=CMFD201802&filename=1018014829.nh&v=MjEwOTVUcldNMUZyQ1VSN3FmWk9kdEZDcmtVYjdJVkYyNkZyTzVHdG5PcHBFYlBJUjhlWDFMdXhZUzdEaDFUM3E=)
[6][Jaemin Son, Sangkeun Kim, Sang Jun Park, Kyu-Hwan Jung:
An Efficient and Comprehensive Labeling Tool for Large-Scale Annotation of Fundus Images. CVII-STENT/LABELS@MICCAI 2018: 95-104](https://dblp.org/pers/s/Son:Jaemin.html)  
[7][Son, J., Park, S.J. & Jung, K. Towards Accurate Segmentation of Retinal Vessels and the Optic Disc in Fundoscopic Images with Generative Adversarial Networks. J Digit Imaging 32, 499–512 (2019). https://doi.org/10.1007/s10278-018-0126-3](https://bitbucket.org/woalsdnd/retinagan/src/master/)  
[8][Fengyan Wang](https://dblp.uni-trier.de/pers/w/Wang:Fengyan.html).  
[9][Lihong Dai, Jinguo Liu, Zhaojie Ju, Yang Gao:
Iris Center Localization Using Energy Map With Image Inpaint Technology and Post-Processing Correction. IEEE Access 8: 16965-16978 (2020)](https://www.sciencedirect.com/science/article/pii/S1361841519301033)


## Challenge3：模型推广
### 问题分析
本题是需要建立模型，将任务一中所完成的任务在广角视网膜图像上同样适用。
![](https://jerrymazeyu.oss-cn-shanghai.aliyuncs.com/2020-03-12-180154.jpg)  
如上图所示，这是一张广角视网膜的眼底图像，图中圈出来的部分就是任务一、二中所用到的图像数据，如何将前面任务中学习到的特征应用到这样完全不同的数据中来，这是一个迁移学习的任务。  
目前，我们常用的迁移学习技术几乎都是对于图像的卷积神经网络（ImageNet）中预训练的参数进行微调已达到迁移学习的目的，在本任务中当然也可以那么做，但是效果肯定不会好，这是因为普通眼底照片的数据集模式是十分统一的，也就是说其学习到的特征如果不加以处理是只能用于常规眼底图像的判别。从统计学的角度来看，这两种图像的分布情况是完全不一样的，任务一的网络学习到的分布根本不适用于任务三。  
但是这个问题同样有一个特点，任务一的图像是任务三的子图，所以普通眼底图像中所有的特征都在广角图像中出现了，那么是不是可以转化成为一个特征识别的问题，例如找到光斑的位置，对其进行周围进行放大，以达到让图片尽可能相似的问题呢？  
基于这种想法，我准备从三个思路去考虑这个问题：  
首先，是从**迁移学习**的角度考虑，尝试利用迁移学习扩展，这其中包含的领域有：基于实例的迁移学习，基于特征的迁移学习。  
其次，是从**特征识别**的角度考虑，当然严格来说也是属于迁移学习的一种，只不过将问题聚焦在**识别光盘**上，通过光盘的位置定位。  
最后，我认为可以从**数据增强**的角度去尝试这个问题，将普通的眼底图进行缩小，并在周围填充一些噪声，有意模拟出一种类似广角视网膜图像的数据集，然后将广角视网膜图像和处理过的普通眼底图像混杂，共同喂入模型进行训练。

### 文献综述与解决方案
对于普通眼底图像模型推广到广角眼底图像的研究，目前并没有发现与之完全对应的论文，于是只能根据问题分析，按照此思路进行文献查阅。  

* 对于**迁移学习**方向，主要有两个重要的名词：**域**和**任务**，域指模型应用的领域，任务主要指下游任务。本题中，显然域和任务是一样的，但是明显分布上面有着很大的差距。[Sinno Jialin](https://www3.ntu.edu.sg/home/sinnopan/publications/TLsurvey_0822.pdf)做了关于迁移学习的综述，其中说明了迁移学习定义和分类；[inFERENCe](https://www.inference.vc/comment-on-overcoming-catastrophic-forgetting-in-nns-are-multiple-penalties-needed-2/)利用贝叶斯学习，提出了通过拉普拉斯近似(Laplace Approximation)的方式来计算贝叶斯公式中那也无法进行数值计算的log-likelihood;[Cross-stitch Networks for Multi-task Learning ](https://arxiv.org/pdf/1604.03539.pdf)这篇文章采用多任务学习，多任务学习是指模型需要把Source和 Target两个或者更多任务同时学好，在本文中也就是说不论何种图像都会进行疾病分类，让这些广角图片和普通图片直接放在一起进行训练；[Wang JinDong](https://github.com/jindongwang/transferlearning)整理了很多关于迁移学习的资料，对于本文任务来说，并不涉及到域的变化，并且迁移学习领域更多讨论的是理论上的可行性，而并不是具体的架构。
* 对于**特征识别**方向，主要是RCNN系列、YOLO系列、以及SSD，由于这些算法都是十分有名的算法，在此不进行赘述，我们可以采用YOLO或者SSD这样的目标检测框架进行训练，包括上文提出的detectron框架也可以做目标检测，而真正的问题是如何对数据进行处理，对于普通的眼底图像来说，得到的标签是一个类别，而如果要训练目标检测的模型，那就必须要对图像进行标注。实际的方法可以是这样，提取中心点是光盘的所有图像，对其中心自动标注一个框，作为光盘的标签。
* 对于**数据增强**方向，主要的研究是根据像素级别的处理进行图像增强，例如[刘国华](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFDLAST2019&filename=1019008696.nh&v=MjEwOTJUM3FUcldNMUZyQ1VSN3FmWk9kc0ZpcmdVcjNOVkYyNkY3TzRGdGZGcVpFYlBJUjhlWDFMdXhZUzdEaDE=)对眼底图像进行相似性处理和多种频谱变换方法避免眼底图像的人工伪影产生，提出了稀疏联合滤波方法，分为两个步骤:初始估计和最终估计。但这并不是本文要做的任务，本文希望做的是数据增强，[He Jun](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=SJES&dbname=SJESLAST&filename=SJES33ABECB4078C31E98C21B1436166AD02&v=MzA4MzVCWk93SGYzODR1aDhibVQxOE9uN21yeFEwZjdUbE1icWRDSlVhRjF1UVVyL1BKbGNTYm1LQ0dZQ0dRbGZCckxVMjU5cGd4cjIyeEtFPU5pZk9mYkM3YjZPNTNQMQ==)提出基于策略梯度的数据增强方法，以在不破坏其内部模式的情况下，在IHC图像中涉及多样性。

综上所述，此问题前人较少研究，本人只能根据问题分析中的设想，进行尝试。首先，可能会先尝试直接对任务一的模型参数fine-tune作为baseline；其次，希望利用眼底图像光盘为中心的特点进行标注，训练目标检测模型，然后再对广角眼底图像进行检测，如果成功就可以将目标区域选中放大，相当于将无关目标区域屏蔽；最后如果效果不好，则会采用上文数据增强的方式实现。不过可以预见的是，对于这个问题，需要写大量的逻辑代码，包括自动标注、选择区域并放大、对普通眼底图进行外围扩充等。

## 难点总结

* 由于我是笔记本电脑，**本地算力严重不足**，可能无法进行大规模训练。
* UNet网络还不是特别熟悉，没有尝试过使用。
* 优化修改网络结构的维度匹配问题。




