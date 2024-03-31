""" 2 Лабораторная работа, input дефолтный из директории """

from PIL import Image
import numpy as np
import os


def semitone(image: Image.Image) -> Image.Image:
    """
    Преобразует цветное изображение в полутоновое(градация серого),
    используя простое арифметическое среднее трех каналов RGB каждого пикселя.
    """
    # noinspection PyTypeChecker
    image_arr = np.array(image.convert('RGB'))  # а то не совсем умный пайчарм...

    # axis=2: ([[[R00, G00, B00]]] -> [[Grey]]), то бишь из 2-d array of RGB arrays мы получаем 2d array with Grey value
    gray_scale = np.mean(image_arr, axis=2)
    new_image = Image.fromarray(gray_scale.astype(np.uint8), 'L')  # 'L' обозначает 8-битную черно-белую картинку
    return new_image


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'input')
    output_path = os.path.join(current_dir, 'output')
    os.makedirs(output_path, exist_ok=True)
    for image in os.scandir(input_path):
        if not image.name.lower().endswith(('.png', '.bmp')):
            print("Ты видимо что-то попутал с:", image.name)
            continue
        print(f"Работаем с {image.name}.")
        output_filepath = os.path.join(output_path, image.name)
        with Image.open(image.path) as output_image:
            semitone(output_image).save(output_filepath)


if __name__ == "__main__":
    main()
