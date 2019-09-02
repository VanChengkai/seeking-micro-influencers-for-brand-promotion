# seeking-micro-influencers-for-brand-promotion

## project description

### abstract
>What made you want to wear the clothes you are wearing? Where is the place you want to visit for your next-coming holiday? Why do you like the music you frequently listen to? If you are like most people, you probably made these decisions as a result of watching influencers on social media. Furthermore, influencer marketing is an opportunity for brands to take advantage of social media using a well-defined and well-designed social media marketing strategy. However, choosing the right influencers is not an easy task. With more people gaining an increasing number of followers in social media, finding the right influencer for an E-commerce company becomes paramount. In fact, most marketers cite it as a top challenge for their brand. To address the aforementioned issues, we propose a data-driven micro-influencer ranking scheme to solve the essential question of finding out the right micro-influencer. Specifically, we represent brands and influencers by fusing their historical posts’ visual and textual information. A novel К -buckets sampling strategy is proposed to learn a brand-micro-influencer scoring function with a modified listwise learning to rank model. In addition, we developed a new Instagram brand micro-influencer dataset, which can benefit future researchers in this area. The extensive evaluation demonstrates the advantage of our proposed method compared with the state-of-the-art methods.

In this project, we proposed a **multi-modal micro-influencer ranking method**, which leverage open data in social media to learn the relevance between brand and micro-influencers. In particular, we design a **social account history pooling mechanism** that leverages both posts' visual and textual content to approximate the semantic representations of a social account. A **modified listwise learning to rank model** is learnt to predict ranking scores for the given brand and micro-influencers. Afterward, we utilize the learnt scoring function to recommend micro-influencers for brand promotion. The extensive evaluation demonstrates the advantage of our proposed method compared with the state-of-the-art methods.<br>

In summary, the contributions of our work are as follows:<br>
* **We design a novel social account history pooling methods, which can leverage social media open data to represent brands and micro-influencers.**
* **We propose a modified listwise learning to rank model, which successfully predict ranking scores for the given brand and micro-influencers. Extensive experiments validate the effectiveness of our proposed model.**
* **We collected and organized a brand-micro-influencer dataset, which can greatly benefit the future researchers in this area.**

Our proposed work consists of two main components: (1) **multi-modal social account representation** and (2) **micro-influencer ranking**. You can see the framework of our model in _figure 1_. We first propose a **novel multimodal social account embedding method**, which exploits social network historical post store present brands and micro-influencers. In particular, we propose a **social account history pooling mechanism** that leverages both posts' visual and textual content to approximate the semantic representations of a social account. We further fuse the visual and textual modalities by a **low-rank bilinear pooling method**. Moreover, we propose a **modified listwise learning to rank model**, and then use this model to predict ranking scores for the given brand and micro-influencers. Specifically, we design a **competence score** for each micro-influencer with respect to a brand as the ranking data label. Then, we modified the listwise learning to rank the model by a **К-buckets sampling strategy**. Afterward, we utilize the learnt scoring function to recommend micro-influencers for brand promotion.<br>

We collect a **social media information dataset named brand micro-influencer dataset**. Please check dataset description for detailed information.<br>

In experiment part, _table 1_ shows the comparison between our proposed **MIR(k)** with the baselines. _Table 2_ shows the comparison between different MIR variants. In order to deep dive into how the proposed method performs on the data in different categories, we set _table 3_ to report the details of results. Furthermore, to have an intuitive understanding of our proposed method, we present some examples of micro-influencer recommendation using the method described in paper. Please check section 5 in our paper for detailed information.<br>
![table1](https://github.com/Mysteriousplayer/seeking-micro-influencers-for-brand-promotion/raw/master/pictures/t01.png "table1")
![table2](https://github.com/Mysteriousplayer/seeking-micro-influencers-for-brand-promotion/raw/master/pictures/t2.png "table2")
![table3](https://github.com/Mysteriousplayer/seeking-micro-influencers-for-brand-promotion/raw/master/pictures/t3.png "table3")
![figure1](https://github.com/Mysteriousplayer/seeking-micro-influencers-for-brand-promotion/raw/master/pictures/f1.png "figure1")


## brand micro-influencer dataset

Here we introduce a **brand micro-influencer dataset** which contains social media information crawled from Instagram. The dataset includes: (1) **360 brand accounts categorized into 12 categories**. (2) **3748 micro-influencer accounts**. (3) **ground-truth for every positive brand micro-influencer pair that can be used for evaluation**. We crawled the latest 1,000 posts of each brand account, and found the candidates list of micro-influencers mentioned within these posts. We further crawled these candidates’ profile pages to retrieve their biographies and the number of followers. From these candidates, we selected **3748 micro-influencer accounts** under the criterion that the number of followers is between _5000_ to _100000_, and removed the non-English accounts. In this way, we paired each brand with around _11_ micro-influencers. Note that there are a small portion (around 10%) of micro-influencers belonging to multiple brands. Based on this dataset, we identify some research issues on micro-influencer marketing. Moreover, we think this dataset can greatly benefit the future researchers in this area.<br>

### Download
>The dataset, including **ground-truth, Brand post information, Micro-influencer post information, Brand post image features, Brand post text features, Micro-influencer post image features, Micro-influencer post text features**. For more detailed descriptions of the dataset, see specification below.<br>
**Brand post information<br>
Micro-influencer post information<br>
Brand post image features<br>
Brand post text features<br>
Micro-influencer post image features<br>
Micro-influencer post text features<br>
Ground-truth**<br>
For any question regarding the dataset, please contact Mr. Shaokun Wang (wangskkk@163.com)

### Citation

### Specification
#### Brand category
>There are 12 categories including Airline, Auto, Clothing, Drink, Electronics, Entertainment, Food, Jewelry, Makeup, Nonprofit, Services, and Shoes. Every category includes 30 brands.<br>

#### Brand post information
>![table4](https://github.com/Mysteriousplayer/seeking-micro-influencers-for-brand-promotion/raw/master/pictures/t4.png "table4")<br>
 **Brand**: brand account user name<br>
 **Post id**: post_id=brand+path1+path2<br>
 **text**: post text information<br>
 **timestamp**: the publish time of a post<br>
 **#comments**: the number of comments in a post<br>
 **#likes**: the number of likes in a post<br>

#### Micro-influencer post information
>![table5](https://github.com/Mysteriousplayer/seeking-micro-influencers-for-brand-promotion/raw/master/pictures/t5.png "table5")<br>
 **Brand**: brand account user name, which is paired with the micro-influencer behind<br>
 **Micro-influencer**: micro-influencer account user name<br>
 **Post id**: post_id=brand+micro-influencer+path1+path2<br>
 **text**: post text information<br>
 **timestamp**: the publish time of a post<br>
 **#comments**: the number of comments in a post<br>
 **#likes**: the number of likes in a post<br>

#### Brand post image features
>We utilize pretrained CNN (VGG16) to extract visual features. Every brand post image feature is 25088-D.

#### Brand post text features
>We utilize Word2Vec to extract textual features. Every brand post text feature is 300-D.

#### Micro-influencer post image features
>We utilize pretrained CNN (VGG16) to extract visual features. Every Micro-influencer post image feature is 25088-D.

#### Micro-influencer post text features
>We utilize Word2Vec to extract textual features. Every Micro-influencer post text feature is 300-D.

#### Ground-truth
