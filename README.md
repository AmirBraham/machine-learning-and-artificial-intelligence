# Music generation with LSTM

Based on this GitHub repository: [Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer)

## Requirements

- Python 3.x
- Install the following packages using pip:
    - Music21
    - Keras
    - TensorFlow
    - h5py

## Model Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 100, 256)          264192

 lstm_1 (LSTM)               (None, 100, 256)          525312

 lstm_2 (LSTM)               (None, 256)               525312

 batch_normalization (Batch  (None, 256)               1024
 Normalization)

 dropout (Dropout)           (None, 256)               0

 dense (Dense)               (None, 256)               65792

 activation (Activation)     (None, 256)               0

 batch_normalization_1 (Bat  (None, 256)               1024
 chNormalization)

 dropout_1 (Dropout)         (None, 256)               0

 dense_1 (Dense)             (None, 110)               28270
_________________________________________________________________
Total params: 1410926 (5.38 MB)
Trainable params: 1409902 (5.38 MB)
Non-trainable params: 1024 (4.00 KB)

```

## Some Thoughts

Create music using a recurrent neural network in Python using the Keras library.

### Dataset

- **MIDI files**: Unlike regular audio files like MP3s or WAVs, MIDI files don't contain actual audio data and are therefore much smaller in size. They explain what notes are played, when they're played, and how long or loud each note should be.

For this tutorial, it’s better to use a single instrument instead of multiple instruments.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*sM3FeKwC-SD66FCKzoExDQ.jpeg" />

Here, instead of “apple,” “orange,” etc., we have “C1,” “A2.C3.E3,” etc.

We concatenate every note/chord together from all MIDI files and use a categorical encoder.

Next, we create input sequences for the network and their respective outputs. The output for each input sequence will be the first note or chord that comes after the sequence of notes in the input sequence in our list of notes.

### Network

- **Loss function**: RMSProp (Root Mean Square Propagation), an adaptive learning rate optimization algorithm designed to address some of the issues encountered with the stochastic gradient descent (SGD) method in training deep neural networks.

[GitHub Repository](https://github.com/Skuldur/Classical-Piano-Composer)[Tutorial](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)

```markdown
# Music Generation with LSTM

Based on this [github repo](https://github.com/Skuldur/Classical-Piano-Composer)

Requirements
Python 3.x

Installing the following packages using pip: Music21 , Keras ,Tensorflow , h5py

Model Architecture 

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 100, 256)          264192    
                                                                 
 lstm_1 (LSTM)               (None, 100, 256)          525312    
                                                                 
 lstm_2 (LSTM)               (None, 256)               525312    
                                                                 
 batch_normalization (Batch  (None, 256)               1024      
 Normalization)                                                  
                                                                 
 dropout (Dropout)           (None, 256)               0         
                                                                 
 dense (Dense)               (None, 256)               65792     
                                                                 
 activation (Activation)     (None, 256)               0         
                                                                 
 batch_normalization_1 (Bat  (None, 256)               1024      
 chNormalization)                                                
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_1 (Dense)             (None, 110)               28270     
...
Total params: 1410926 (5.38 MB)
Trainable params: 1409902 (5.38 MB)
Non-trainable params: 1024 (4.00 KB)
```

Create music using a recurrent neural network in Python using the Keras library.

Dataset ? 

Midi files

Unlike regular audio files like [MP3s](https://www.lifewire.com/mp3-file-2622780) or [WAVs](https://www.lifewire.com/wav-wave-files-2622395), these don't contain actual audio data and are therefore much smaller in size. They instead explain what notes are played, when they're played, and how long or loud each note should be.

https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

It’s better for this tutorial to use a single instrument instead of multiple instruments ( will test it out )

![Untitled](Music%20generation%20with%20LSTM%207e34e24af9d5453da2d2d64f109f16e6/Untitled.png)

Here instead of “apple”,”orange”.. we have “C1”,”A2.C3.E3”..

We concatenate every note/chord together from all midis and use a categorical encoder 

Next, we have to create input sequences for the network and their respective outputs. The output for each input sequence will be the first note or chord that comes after the sequence of notes in the input sequence in our list of notes

Network :

Loss function ? 

RMSProp, which stands for **Root Mean Square Propagation**, is an adaptive learning rate optimization algorithm designed to address some of the issues encountered with the stochastic gradient descent (SGD) method in training deep neural networks.

The proposed architecture uses a fixed offset of 0.5 between different notes. 

How to encode timing between notes ?
