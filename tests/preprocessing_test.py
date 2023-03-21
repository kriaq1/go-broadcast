import src.preprocessing as preprocess
import cv2


def preprocess_image(image_path='preprocessing_data/1.jpg', save_path='preprocessing_data/'):
    image = cv2.imread(image_path)

    cv2.imwrite(save_path + "6.rotate" + '.jpg', preprocess.rotate(image))

    image = preprocess.padding(image, 1024)

    cv2.imwrite(save_path + "1.thresholding" + '.jpg', preprocess.thresholding(image, 125))

    cv2.imwrite(save_path + "2.adaptive_thresholding" + '.jpg', preprocess.adaptive_thresholding(image))

    cv2.imwrite(save_path + "3.canny" + '.jpg', preprocess.canny(image))

    cv2.imwrite(save_path + "5.padding" + '.jpg', preprocess.padding(image, 512))

    cv2.imwrite(save_path + "4.shadows" + '.jpg', preprocess.remove_shadows(image))

    cv2.imwrite(save_path + "7.shift" + '.jpg', preprocess.shift(image, -250, 250))


if __name__ == '__main__':
    preprocess_image()
