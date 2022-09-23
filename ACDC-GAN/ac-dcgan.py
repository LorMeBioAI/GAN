import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, MaxPooling2D, Conv2D, Conv2DTranspose, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils import to_categorical
from pic_preprocess import PicPreProcessing

# K.set_image_data_format('channels_first')
K.set_image_data_format('channels_last')


def processingData():
    path1 = 'prepare/blue1'
    avatar = PicPreProcessing()
    imgs = []
    labels = []
    blue_name_list = avatar._get_img_list(path1)
    blue = [avatar._get_img(name) for name in blue_name_list]
    imgs.extend(np.array(blue))
    label = [0 for _ in range(len(blue_name_list))]
    labels.extend(label)
    for i in range(1, 6):
        name_list = avatar._get_img_list('prepare/fig' + str(i))
        img = [avatar._get_img(name) for name in name_list]
        label = [i for z in range(len(name_list))]
        imgs.extend(np.array(img))
        labels.extend(label)
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    # X_train = X_train[:, np.newaxis, :, :]
    # y_train = to_categorical(y_train)
    imgs = np.array(imgs)
    labels = to_categorical(labels)
    # labels = np.array(labels)

    return imgs, labels


def Generator():
    generator = Sequential(name='generator')
    # Transforms the input into a 7 × 7 128-channel feature map
    # generator.add(Dense(128 * 7 * 7, input_dim=latent_dim))
    generator.add(Dense(16 * 16 * 1, input_dim=latent_dim))
    generator.add(Dense(512 * 4 * 4, input_dim=512))
    generator.add(Dense(4 * 4 * 1024, input_dim=8192))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((4, 4, 1024)))
    generator.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())
    generator.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())
    generator.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())
    generator.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(LeakyReLU(0.2))
    print(generator.summary())
    generator.compile(loss='binary_crossentropy', optimizer=adam)
    return generator


# Make Discriminator Model
def Discriminator():
    discriminator = Sequential(name='discriminator')
    # discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
    #                          input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same',
                             input_shape=(64, 64, 3), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid', name="dis_output"))
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    print('dis', discriminator.summary())

    return discriminator


# Make Classifier Model
def Classifier():
    classifier = Sequential(name='classifier')
    classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                          input_shape=(64, 64, 3)))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(6, activation='softmax', name="class_output"))

    print(classifier.name)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print('cls', classifier.summary())

    return classifier


# Creating the Adversarial Network. We need to make the Discriminator weights
# non trainable. This only applies to the GAN model.

def GAN():
    classifier.trainable = False
    discriminator.trainable = False

    ganInput = Input(shape=(latent_dim,))
    x = generator(ganInput)
    gan = Model(inputs=ganInput, outputs=[discriminator(x), classifier(x)], name='gan')
    losses = {
        discriminator.name: "binary_crossentropy",
        classifier.name: "categorical_crossentropy",
    }
    lossWeights = {discriminator.name: 1.0, classifier.name: 1.0}
    gan.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)
    print('gan', gan.summary())
    return gan


# Plot the loss from each batch
def plotLoss(epoch, dLosses, gLosses, cLosses):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.plot(cLosses, label='Classifier loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)


def plotGeneratedImages(epoch, examples=60, dim=(6, 10), figsize=(6, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim - 6])
    print('noise.shape', noise.shape)
    labels = None
    for i in range(6):
        for j in range(10):
            if labels is None:
                labels = np.array([[int(i == k) for k in range(6)]])
            else:
                labels = np.concatenate((labels, np.array([[int(i == k) for k in range(6)]])), axis=0)
    print(labels.shape)
    noise = np.concatenate((noise, labels), axis=1)

    generatedImages = generator.predict(noise)
    # np.save('generatedImages.npy',generatedImages)
    # generatedImages = generatedImages.astype('uint8')

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        # cv2.imshow('imshow',generatedImages[i])
        newimg = generatedImages[i] * 250.0
        typenew = newimg.astype('uint8')
        plt.imshow(typenew, cmap='Blues')

        # cv2.waitKey(0)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/blue_epoch_%d.png' % epoch)


# Save the generator and discriminator networks (and weights) for later use
# def saveModels(epoch):
#     generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
#     discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)
#     classifier.save('models/dcgan_classifier_epoch_%d.h5' % epoch)


"""## Train our GAN and Plot the Synthetic Image Outputs 
After each consecutive Epoch we can see how synthetic images being improved
"""


def train():
    gan = GAN()
    epochs = 10000
    batchSize = 512
    batchCount = X_train.shape[0] / batchSize

    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    dLosses = []
    gLosses = []
    cLosses = []
    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for i in tqdm(range(int(batchCount))):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, latent_dim])
            random_index = np.random.randint(0, X_train.shape[0], size=batchSize)
            imageBatch = X_train[random_index]
            labels = y_train[random_index]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            print('imageBatch', imageBatch.shape)
            print('generatedImages', generatedImages.shape)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2 * batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train Classifier
            classifier.trainable = True
            closs, _ = classifier.train_on_batch(np.concatenate([imageBatch]), labels)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, latent_dim - 6])
            noise = np.concatenate((noise, labels), axis=1)
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            classifier.trainable = False
            gloss = gan.train_on_batch(noise, {discriminator.name: yGen, classifier.name: labels})

            # Store loss of most recent batch from this epoch
            dLosses.append(dloss)
            gLosses.append(gloss)
            cLosses.append(closs)

        if e == 1 or e % 100 == 0:
            # Plot losses from every epoch
            plotGeneratedImages(e)
            plotLoss(e, dLosses, gLosses, cLosses)
            # saveModels(e)


if __name__ == '__main__':
    X_train, y_train = processingData()
    # processingData()
    adam = Adam(lr=0.0005, beta_1=0.5)
    latent_dim = 100 + 6
    generator = Generator()
    discriminator = Discriminator()
    classifier = Classifier()
    train()
