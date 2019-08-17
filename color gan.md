# Image Colorization with GANs
This is implementation of the paper **Image Colorization with Generative Adversarial Networks**. I have implementated the above paper in keras.

[Paper Link](https://arxiv.org/abs/1803.05400)

The process of automatic colorization of black and white images has been a significant area of research for many people and also it has applications in many areas. The problem is higly ill posed as due high degree of freedoms during the assignment of color information. Some methods for this kind of problem involved autoencoders. The paper which is implemented here suggests the use of Generative Adversarial Networks for this task. GANs have been a great research in the recent yeras and they have been widely used in lots of tasks like in this case the task of image colorization .

I have used colornet dataset which is public dataset on FloydHub. A total of 500 images. The model was trained for 200 epochs. Although the writers of original reseach paper used cifar-32 dataset having 1.8 million images and on Places365 dataset having 50,000 images. I was not able to do training on such large dataset due to the non availability of high end hardwares for training on such large datasets. That is why the results are also not that good as compare to original paper. If it was trained on such large datasets the results would have been better.

**Model Architecture**

![model](color_gan_model.png)

## Results
As I menetioned earlier I used small dataset and train the model on low epochs. So my results were not as good as in the original paper. Some of the good results are here.

I trained the model first by the methods and hyperparameter mentioned in original paper. The model was trained for 100 epochs and learning rate was decreased manually after 82 epochs as loss was saturating was saturating. The results were not that good.

Then I trained the model by doing little changes in hyperparameter and in model and then the results on training images were very good and oon test images were not promising but i assumed it was because the traing data was low and the model was trained for less epochs, had there been more training data the results on test images would have been better. 



These are traing images result for model trained on the basis of original paper.

**On training data**

grey scale image---------------model colorized image-----------actual color image

![black](images/original/ac_black.png) ![color](images/original/ac_color.png) ![actual](images/original/ac_actual.png) 



The following are the results of model trained by new hyperparametrs and doing a little change in model

**On training data**

![black](images/new/black_new.png) ![color](images/new/color_new.png) ![actual](images/new/actual_new.png) 

**On test data**

![black](images/new/blackte_new.png) ![color](images/new/colorte_new.png) ![actual](images/new/actualte_new.png) 

