import os
from PIL import Image
from typing import Callable
import numpy as np


def two_step_resampling(img: np.array) -> np.array:
    numerator = int(input('Введите целый коэффициент растяжения:\n> '))
    denominator = int(input('Введите целый коэффициент сжатия:\n> '))
    tmp = one_step_resampling(img, numerator,
                              dimension_func=lambda a, b: a * b,
                              create_img_func=lambda a, b: round(a / b))
    raw_result = one_step_resampling(
        img=tmp,
        factor=denominator,
        dimension_func=lambda a, b: round(a / b),
        create_img_func=lambda a, b: a * b
    )

    return Image.fromarray(raw_result.astype(np.uint8), 'RGB')


def one_step_resampling(img: np.array, factor, dimension_func, create_img_func) -> np.array:
    """ dimension_func, create_img_func - абстракции для вычисления новых размеров и выбора соответствующих пикселей"""
    dimensions = img.shape[0:2]
    new_dimensions = tuple(dimension_func(dimension, factor) for dimension in dimensions)
    new_shape = (*new_dimensions, img.shape[2])
    new_img = np.empty(new_shape)

    for x in range(new_dimensions[0]):
        for y in range(new_dimensions[1]):
            new_img[x, y] = img[
                min(create_img_func(x, factor), dimensions[0] - 1),
                min(create_img_func(y, factor), dimensions[1] - 1)
            ]
    return new_img


def one_step_wrapper(img, var_type, f1, f2):
    factor = var_type((input(f'Введите {"целое число" if var_type == int else "дробное число"}:\n> ')))
    result = Image.fromarray(one_step_resampling(img, factor, f1, f2).astype(np.uint8), 'RGB')
    return result


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output')
    input_path = os.path.join(current_dir, 'input')  # 1sem/1/input/*
    print("""Операции:
    1) Растяжение (интерполяция) изображения в M раз;
    2) Сжатие(децимация) изображения в N раз;
    3) Передискретизация изображения в K=M/N раз путём растяжения и
    последующего сжатия (в два прохода);
    4) Передискретизация изображения в K раз за один проход.
    """)
    for image in os.scandir(input_path):
        print(f"Работаем с {image.name}.")
        # noinspection PyTypeChecker
        img_np = np.array(Image.open(image.path).convert('RGB'))  # а то не совсем умный пайчарм...

        operation: Callable[[np.ndarray], Image.Image] = {
            # K - float, M и N - int
            # Растяжение (интерполяция) изображения в M раз;
            1: lambda img: one_step_wrapper(img, int,
                                            lambda a, b: a * b,
                                            lambda a, b: round(a / b)),  # ближайший сосед
            # Сжатие (децимация) изображения в N раз;
            2: lambda img: one_step_wrapper(img, int,
                                            lambda a, b: round(a / b),
                                            lambda a, b: a * b),
            # Передискретизация изображения в K=M/N раз путём растяжения и последующего сжатия (в два прохода);
            3: lambda img: two_step_resampling(img),
            # Передискретизация изображения в K раз за один проход
            4: lambda img: one_step_wrapper(img, float,
                                            lambda a, b: round(a * b),
                                            lambda a, b: round(a / b))
        }.get(int(input('Выберите операцию:\n> ')))
        if operation:
            operation(img_np).save(os.path.join(output_path, image.name))
            print('Сохранили изображение!\n\n')
        else:
            print('Вводить нужно было число от 1 до 4, ты что не знал?)')
            exit()


if __name__ == "__main__":
    """ 1 Лабораторная работа """
    main()
