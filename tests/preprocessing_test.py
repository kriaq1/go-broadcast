import src.preprocessing as preprocess
import cv2


def preprocess_image(image_path='PreprocessingData/1.jpg', save_path='PreprocessingData/Result/'):
    image = cv2.imread(image_path)

    cv2.imwrite(save_path + "thresholding" + '.jpg', preprocess.thresholding(image))

    cv2.imwrite(save_path + "adaptive_thresholding" + '.jpg', preprocess.adaptive_thresholding(image))

    cv2.imwrite(save_path + "padding" + '.jpg', preprocess.padding(image, 1024))

    cv2.imwrite(save_path + "shadows" + '.jpg', preprocess.remove_shadows(image))


preprocess_image()
