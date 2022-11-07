
#Unet Explaination

U-net is a type of convolutional network that makes use of fewer high quality training images to better and faster identify object in images given. U-net gets its name for its 'U' shaped network architechture (See below).

<img width="699" alt="Unet network architecture" src="https://user-images.githubusercontent.com/59149625/200168269-483b4e00-595d-438f-af33-ddb1834bb1b1.png">

The U-net architecure is made up of a contracting path on the left and expanding path on the right as seen above. These contracting and expanding paths gives the architecture its name. The contacting path is similar to most convolutional networks, where convulation happen twice of size 3x3. It then passed to the ReLU activation function. At this point the network uses max pooling to find the most prominent part of the segmented image and outputs a 2x2 max pool. The expanding path does the reverse instead of breaking the image down it builds it back up using upsampling 2x2 convoluation. It is important to crop images and create tiles such that 2x2 max pooling can be done cleanly allowing for better output segmentations.

U-net architecutre is useful when there is a lack of training data such as in the medical field. For most models little training data would result in models that are not good. U-net architecutre can trained with randomly deformed input images. This is good because it allows us to make good predictions when given little access to data like those in the medical field.
