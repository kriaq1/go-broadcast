import cv2


def padding(image, size=1024, inter=cv2.INTER_AREA):
    old_height, old_width = image.shape[0], image.shape[1]
    if old_height <= old_width:
        new_width = size
        new_height = size * old_height // old_width
        image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        y = size - image.shape[0]
        image = cv2.copyMakeBorder(image, y // 2, y // 2 + y % 2, 0, 0, cv2.BORDER_CONSTANT)
    else:
        new_width = size * old_width // old_height
        new_height = size
        image = cv2.resize(image, (new_width, new_height), interpolation=inter)
        x = size - image.shape[1]
        image = cv2.copyMakeBorder(image, 0, 0, x // 2, x // 2 + x % 2, cv2.BORDER_CONSTANT)
    return image
