# import train
import predict

if __name__ == '__main__':
    # train.train_autoencoder(epoch_plus=155)
    # test.test_autoencoder(epoch_plus=400)
    dir = '/home/sgvr/wkim97/EE474/ee474/data/test_images/18.png'
    predict.predict_image(dir)
