
# Unet Explaination

U-net is a type of convolutional network that makes use of fewer high quality training images to better and faster identify object in images given. U-net gets its name for its 'U' shaped network architechture (See below).

<img width="699" alt="Unet network architecture" src="https://user-images.githubusercontent.com/59149625/200168269-483b4e00-595d-438f-af33-ddb1834bb1b1.png">

The U-net architecure is made up of a contracting path on the left and expanding path on the right as seen above. These contracting and expanding paths gives the architecture its name. The contacting path is similar to most convolutional networks, where convulation happen twice of size 3x3. It then passed to the ReLU activation function. At this point the network uses max pooling to find the most prominent part of the segmented image and outputs a 2x2 max pool. The expanding path does the reverse instead of breaking the image down it builds it back up using upsampling 2x2 convoluation. It is important to crop images and create tiles such that 2x2 max pooling can be done cleanly allowing for better output segmentations.

U-net architecutre is useful when there is a lack of training data such as in the medical field. For most models little training data would result in models that are not good. U-net architecutre can trained with randomly deformed input images. This is good because it allows us to make good predictions when given little access to data like those in the medical field.

# A: Segmented Images from the Validation Set
![seg_img_1](https://user-images.githubusercontent.com/59149625/200174400-3f88c26f-37f2-4e8c-a03a-fc289790b9d9.png)
![seg_img_2](https://user-images.githubusercontent.com/59149625/200174401-96174c85-ec66-4fd7-a88a-84bfd82d0645.png)
![seg_img_3](https://user-images.githubusercontent.com/59149625/200174402-3c50f75a-4945-4098-813f-57153cb925ce.png)
![seg_img_4](https://user-images.githubusercontent.com/59149625/200174403-d3dfeb3a-697e-4971-b63d-6c4e3ea6ca11.png)
![seg_img_5](https://user-images.githubusercontent.com/59149625/200174405-0435827f-17fa-4fe8-8327-6e1a462f6a7f.png)
![seg_img_6](https://user-images.githubusercontent.com/59149625/200174406-6c1fe181-c0d1-4419-aaf5-abda03c19950.png)
![seg_img_7](https://user-images.githubusercontent.com/59149625/200174407-3b05ff41-1dcc-4a54-bb8e-5020d03777ea.png)
![seg_img_8](https://user-images.githubusercontent.com/59149625/200174408-2a4c81e7-bbc6-4791-9a4c-ccccf2ba469e.png)
![seg_img_9](https://user-images.githubusercontent.com/59149625/200174409-31f8a198-8f46-4be7-b861-b9ee81ae618b.png)


# B: Training and Validation Loss vs Epochs
![Training_Validation_Loss_Epochs](https://user-images.githubusercontent.com/59149625/200174456-0b2c4758-730f-4abd-8f90-fce4eb8d21e0.PNG)

# C: Precision and Recall values
![Precision and Recall Values](https://user-images.githubusercontent.com/59149625/200174486-8c1eec5b-250e-4f35-b414-77fe48cbab2d.PNG)
