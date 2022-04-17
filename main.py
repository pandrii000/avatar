import cv2
import numpy as np
import argparse


def hex2rgb(h: str):
    return tuple(int(h[i: i + 2], 16) for i in (1, 3, 5))


def crop_center_square(img):
    H, W = img.shape[:2]
    size = min(H, W)

    yc = H // 2
    xc = W // 2
    if H < W:
        img = img[:, xc - size // 2: xc - size // 2 + size]
    else:
        img = img[yc - size // 2: yc - size // 2 + size, :]

    return img


def avatar(input_path: str, color: str, output_path: str):
    img = cv2.imread(input_path)

    img = crop_center_square(img)

    size = min(img.shape[:2])
    radius = size // 2 - round(size * 0.1)

    ins_mask = cv2.circle(np.zeros_like(img) + 255, (size // 2, size // 2), radius, (0, 0, 0), -1)
    out_mask = cv2.circle(np.zeros_like(img), (size // 2, size // 2), radius, (255, 255, 255), -1)

    img = cv2.bitwise_and(img, out_mask)

    back = (np.zeros_like(img) + hex2rgb(color)).astype('uint8')
    back = cv2.bitwise_and(back, ins_mask)

    img = img + back

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./input/photo.jpg')
    parser.add_argument('--color', default='#5B4B49')
    parser.add_argument('--output', default='./output/avatar.jpg')
    args = parser.parse_args()

    avatar(args.input, args.color, args.output)
