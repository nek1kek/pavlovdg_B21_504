""" 4 Лабораторная работа """
import dataclasses
import os
import numpy as np
from PIL import Image

from tqdm import tqdm

G_x = np.array([[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]])
G_y = np.array([[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]])

WHITE, BLACK = 255, 0


@dataclasses.dataclass
class ImageHelper:
    def __init__(self, read_image: Image.Image, image_dir: str):
        self.image_dir = image_dir
        # noinspection PyTypeChecker
        self.image_arr = np.array(read_image.convert('L')).astype(np.uint8)  # easiest semitone ITU-R 601-2
        self.new_img_g_x = np.zeros_like(self.image_arr, dtype=np.float64)
        self.new_img_g_y = np.zeros_like(self.image_arr, dtype=np.float64)
        self.new_img_g = np.zeros_like(self.image_arr, dtype=np.float64)
        self.new_img_g_bin = np.zeros_like(self.image_arr, dtype=np.float64)

    def flush(self):
        os.makedirs(self.image_dir, exist_ok=True)
        Image.fromarray(self.new_img_g_x.astype(np.uint8), 'L').save(os.path.join(self.image_dir, "g_x.png"))
        Image.fromarray(self.new_img_g_y.astype(np.uint8), 'L').save(os.path.join(self.image_dir, "g_y.png"))
        Image.fromarray(self.new_img_g.astype(np.uint8), 'L').save(os.path.join(self.image_dir, "g.png"))
        self.new_img_g_bin = np.where(self.new_img_g > BINARIZATION_TREHOLD, WHITE, BLACK)
        Image.fromarray(self.new_img_g_bin.astype(np.uint8), 'L').save(os.path.join(self.image_dir, "bin.png"))


def sharr_operator(img: ImageHelper) -> None:
    rows, columns = img.image_arr.shape
    for row in tqdm(range(1, rows - 1), desc="линий файла"):
        for col in range(1, columns - 1):
            frame = img.image_arr[row - 1: row + 2, col - 1:col + 2]

            gradient_x = np.sum(G_x * frame.astype(np.int32))
            img.new_img_g_x[row, col] = gradient_x

            gradient_y = np.sum(G_y * frame.astype(np.int32))
            img.new_img_g_y[row, col] = gradient_y

            img.new_img_g[row, col] = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # нормализируем до отенков серого
    img.new_img_g_x *= WHITE / np.max(img.new_img_g_x)
    img.new_img_g_y *= WHITE / np.max(img.new_img_g_y)
    img.new_img_g *= WHITE / np.max(img.new_img_g)

    img.flush()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    input_path = os.path.join(current_dir, 'input')

    os.makedirs(output_path, exist_ok=True)
    for image in os.scandir(input_path):
        if not image.name.lower().endswith(('.png', '.bmp')):
            print("Ты видимо что-то попутал с:", image.name)
            continue
        with Image.open(image.path) as read_image:
            print(f"Работаем с {image.name}.")
            image_dir = os.path.join(output_path, image.name)
            sharr_operator(ImageHelper(read_image, image_dir))


if __name__ == "__main__":
    BINARIZATION_TREHOLD = 30  # лень...
    main()
