# Urban Sound Classification
### Deep Learning (Convolutional Neural Network)

![](https://static.wikia.nocookie.net/bleach/images/1/16/Ep329UraharaProfileOption4.png/revision/latest/scale-to-width-down/1000?cb=20220325000742&path-prefix=en)

## Libraries
- Numpy 
- Matplotlib 
- Pandas 
- Librosa 
- Tensorflow.keras
- sklearn
## Dataset Analysis
This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:  
| 0 = airconditioner  | 5 = engineidling |
|---------------------|------------------|
| 1 = carhorn         | 6 = gunshot      |
| 2 = childrenplaying | 7 = jackhammer   |
| 3 = dogbark         | 8 = siren        |
| 4 = drilling        | 9 = street_music |


We've used librosa to generate grayscale heatmap of the data.  
![](https://i.ibb.co/0hQ45Mq/indir.png)
## Spectrogram
We've generated spectrogram for each data with librosa.  
~~~~
def create_pectrogram(path):
  y, sr = librosa.load(path)
  #spec_conv = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)
  spec = librosa.feature.melspectrogram(y=y)
  spec_conv = librosa.amplitude_to_db(spec, ref = np.max)
  spec_mean = np.mean((spec_conv / 80.0).T, axis = 0)
  return spec_mean
~~~~
~~~~
spectrogram = []
classid = []

for i in range(data.shape[0]):
  file_name = "/content/UrbanSound/UrbanSound8K/audio/fold" + str(data["fold"][i]) + "/" + data["slice_file_name"][i]
  label = data["classID"][i]

  spec_conv = create_pectrogram(file_name)

  spectrogram.append(spec_conv)
  classid.append(label)
~~~~
![](https://i.ibb.co/0XGy0fn/spec.png)
## Preprocessing
We need to use (X,128) data shape in order to use Convolutional Neural Network. So we are shaping our data as (X.shape,128).  
~~~~
X_ = data_last[0]
Y = data_last[1]
X = np.empty([data.shape[0], 128])
~~~~
## Convolutional Neural Network Implementation
Now we need to arrange our model to get input as much as our data has and output as 1 because our data has only one class. And we used 2 hidden layers, first one is getting 16 inputs, while second is getting 8. Making 128 total as Convolutional Neural Network needs.
~~~~
X_train = X_train.reshape(X_train.shape[0], 16, 8, 1)
X_test = X_test.reshape(X_test.shape[0], 16, 8, 1)
~~~~
**Here our Convolutional Neural Network Model:**  
  
![](https://i.ibb.co/YyczCnf/Untitled.png)
## Results
The following results are obtained by using 0.25 train to test ratio.  
  
Test Accuracy: 90.24%  
Test Loss: 63.17%

Here is the graphical accuracy result:  
![](https://i.ibb.co/2NwVMpH/indir-1.png)  
Here is the graphical loss result:  
![](https://i.ibb.co/V3zJy34/indir-2.png)  
## References
1. [Global AI Hub](https://globalaihub.com/courses/introduction-to-deep-learning/)
2. [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
3. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sAiXy-QZhKwimZfcQAQ34dMw5N9TN26y?usp=sharing)
---
