import os
import numpy as np
from PIL import Image
from tqdm import tqdm

BLACK, WHITE = 0, 255
OMEGA_OF_WHITE = 8 * WHITE


def logical_filtering(image_arr):
    res = np.zeros_like(image_arr, dtype=np.uint8)

    rows, columns = image_arr.shape

    # стенки втупую проставим в округ
    res[0] = image_arr[0]
    res[~0] = image_arr[~0]
    for row in range(rows):
        res[row, 0] = image_arr[row, 0]
        res[row, -1] = image_arr[row, -1]

    for row in tqdm(range(1, rows - 1), desc="линий файла"):
        for col in range(1, columns - 1):

            big_omega = sum([image_arr[row + i, col + j] for i in range(-1, 2, 2) for j in range(-1, 2, 2)])
            if big_omega == BLACK and image_arr[row, col] == WHITE:
                # пустышки меня окружают
                res[row, col] = BLACK
            elif big_omega == OMEGA_OF_WHITE and image_arr[row, col] == BLACK:
                # кругом однерки
                res[row, col] = WHITE
            else:
                res[row, col] = image_arr[row, col]
    return res


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
            # noinspection PyTypeChecker
            image_arr = np.array(read_image, np.uint8)

            res_arr = logical_filtering(image_arr)
            Image.fromarray(res_arr, 'L').save(os.path.join(output_path, f"filtered_{image.name}"))

            diff_arr = image_arr ^ res_arr
            Image.fromarray(diff_arr, 'L').save(os.path.join(output_path, f"xor_{image.name}"))


if __name__ == "__main__":
    main()
