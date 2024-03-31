from PIL import Image
import numpy as np
import os
import math
from tqdm import tqdm

MIN_DIFF_LIST = [5, 10, 15]  # epsilen в презе
SMALL_WINDOW_SIZE = 5
BIG_WINDOW_SIZE = 15
BLACK = 0
WHITE = 255


def otsu_local_treshold(window):
    # CHECK PRESENTATION TO UNDERSTAND WHAT HAPPENED, U WROTE IT WHILE GETTING HIGH, U CAN`T REMEMBER
    bins = np.arange(np.min(window) - 1, np.max(window) + 1)
    hist, base = np.histogram(window, bins=bins, density=True)
    base = base[1:].astype(np.uint8)
    w_0 = np.cumsum(hist)
    t_rank = 0
    i_max = 0
    i = -1
    for w0 in w_0:
        i += 1

        m_0 = np.sum(base[:i] * hist[:i] / w0)
        m_1 = np.sum(base[i + 1:] * hist[i + 1:] / (1 - w0))

        d_0 = np.sum(hist[:i] * (base[:i] - m_0) ** 2)
        d_1 = np.sum(hist[i + 1:] * (base[i + 1:] - m_1) ** 2)

        d_all = w0 * d_0 + (1 - w0) * d_1
        d_class = w0 * (1 - w0) * (m_0 - m_1) ** 2

        if d_all == 0:
            i_max = i
            break
        if d_class / d_all > t_rank:
            t_rank = d_class / d_all
            i_max = i

    return base[i_max]


def get_means(matrix, t):
    values = matrix.flatten()
    mean_func = lambda x: x.mean() if x.size else 0
    return mean_func(values[values >= t]), mean_func(values[values < t])


def pixel_change(small_window, big_window, min_diff):
    t = otsu_local_treshold(big_window)
    up_mean, less_mean = get_means(big_window, t)
    if math.fabs(less_mean - up_mean) >= min_diff:
        new_image_small_window = np.zeros(small_window.shape)
        new_image_small_window[small_window > t] = WHITE
        return new_image_small_window
    small_window_mean = small_window.mean()
    if math.fabs(less_mean - small_window_mean) < math.fabs(up_mean - small_window_mean):
        return np.full(small_window.shape, WHITE)


def get_big_window(image_arr, w, h):
    up_row = max(h - BIG_WINDOW_SIZE // 2 + 1, 0)
    down_row = min(h + BIG_WINDOW_SIZE // 2 + SMALL_WINDOW_SIZE - 1, image_arr.shape[0])
    left = max(w - BIG_WINDOW_SIZE // 2 + 1, 0)
    right = min(w + BIG_WINDOW_SIZE // 2 + SMALL_WINDOW_SIZE - 1, image_arr.shape[1])
    return image_arr[up_row:down_row, left:right]


def Eikvel_binarization(image, min_diff):
    image_arr = np.mean(np.array(image), axis=2).astype(np.uint8)  # semitone из 1 части лабы
    new_image_arr = np.zeros(shape=image_arr.shape)
    np.full((2, 2), WHITE).astype(np.uint8)
    for h in tqdm(range(0, image_arr.shape[0], SMALL_WINDOW_SIZE)):
        for w in range(0, image_arr.shape[1], SMALL_WINDOW_SIZE):
            big_window = get_big_window(image_arr, w, h)
            small_window = image_arr[h:h + SMALL_WINDOW_SIZE, w:w + SMALL_WINDOW_SIZE]
            new_small_window = pixel_change(small_window, big_window, min_diff)
            new_image_arr[h:h + SMALL_WINDOW_SIZE, w:w + SMALL_WINDOW_SIZE] = new_small_window

    return Image.fromarray(new_image_arr.astype(np.uint8), 'L')


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    input_path = os.path.join(current_dir, 'input')

    os.makedirs(output_path, exist_ok=True)
    for min_diff in MIN_DIFF_LIST:
        var_dir_path = os.path.join(output_path, f'min_diff={min_diff}')
        os.makedirs(var_dir_path, exist_ok=True)
        for image in os.scandir(input_path):
            if not image.name.lower().endswith(('.png', '.bmp')):
                print("Ты видимо что-то попутал с:", image.name)
                continue
            with Image.open(image.path) as read_image:
                print(f"Работаем с {image.name} with min_diff={min_diff}.")
                binarized_image = Eikvel_binarization(read_image, min_diff)
                binarized_image.save(os.path.join(var_dir_path, image.name))


if __name__ == "__main__":
    main()
