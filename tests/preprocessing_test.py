import src.preprocessing as preprocess
import cv2


def preprocess_image(image_path='PreprocessingData/1.jpg', save_path='PreprocessingData/'):
    image = cv2.imread(image_path)

    image = preprocess.padding(image)

    cv2.imwrite(save_path + "1.thresholding" + '.jpg', preprocess.thresholding(image, 125))

    cv2.imwrite(save_path + "2.adaptive_thresholding" + '.jpg', preprocess.adaptive_thresholding(image))

    cv2.imwrite(save_path + "4.padding" + '.jpg', preprocess.padding(image, 512))

    cv2.imwrite(save_path + "3.shadows" + '.jpg', preprocess.remove_shadows(image))

    cv2.imwrite(save_path + "5.rotate" + '.jpg', preprocess.rotate(image))

    cv2.imwrite(save_path + "6.shift" + '.jpg', preprocess.shift(image, -250, 250))


preprocess_image()
