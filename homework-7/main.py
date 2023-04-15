import cv2
import numpy as np

def ideal_high_pass_filter(image, cutoff_frequency):
    # Преобразование изображения в полутоновое
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение преобразования Фурье
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    # Создание маски для идеального фильтра высоких частот
    rows, cols = gray_image.shape
    mask = np.ones((rows, cols), np.uint8)
    mask[int(rows/2) - cutoff_frequency:int(rows/2) + cutoff_frequency,
         int(cols/2) - cutoff_frequency:int(cols/2) + cutoff_frequency] = 0

    # Применение маски и обратное преобразование Фурье
    fshift_filtered = fshift * mask
    f_filtered = np.fft.ifftshift(fshift_filtered)
    img_filtered = np.fft.ifft2(f_filtered)
    img_filtered = np.abs(img_filtered)

    # Нормализация изображения для отображения
    img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return img_filtered

def butterworth_highpass_filter(image, d0=30, n=2):
    h, w = image.shape[:2]

    # Преобразование Фурье
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Создание маски фильтра Баттерворта
    x, y = np.meshgrid(np.arange(-w//2, w//2), np.arange(-h//2, h//2))
    distance = np.sqrt(x**2 + y**2)
    mask = 1 / (1 + (d0 / distance)**(2*n))

    # Применение фильтра к изображению
    filtered_shift = fshift * mask

    # Обратное преобразование Фурье
    filtered = np.fft.ifftshift(filtered_shift)
    filtered_image = np.fft.ifft2(filtered)
    filtered_image = np.abs(filtered_image)

    return filtered_image

def gauss_high_pass_filter(image, d0):
    # Получаем размеры изображения
    h, w = image.shape[:2]

    # Переводим изображение в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем преобразование Фурье
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)

    # Создаем фильтр H(u, v)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    u = x - w // 2
    v = y - h // 2
    d_squared = u ** 2 + v ** 2
    h_filter = 1 - np.exp(-d_squared / (2 * d0 ** 2))

    # Применяем фильтр к преобразованию Фурье
    filtered_fshift = fshift * h_filter

    # Применяем обратное преобразование Фурье
    ifshift = np.fft.ifftshift(filtered_fshift)
    filtered_image = np.fft.ifft2(ifshift)

    # Округляем значения и переводим в uint8
    filtered_image = np.abs(filtered_image)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image


if __name__ == '__main__':
    # Ideal filter
    image = cv2.imread('car.jpeg')
    img_filtered15 = ideal_high_pass_filter(image, 15)
    img_filtered30 = ideal_high_pass_filter(image, 30)
    img_filtered80 = ideal_high_pass_filter(image, 80)
    cv2.imwrite('car_ideal_result15.jpg', img_filtered15)
    cv2.imwrite('car_ideal_result30.jpg', img_filtered30)
    cv2.imwrite('car_ideal_result80.jpg', img_filtered80)

    
    # Butterworth
    image_path = "roses.jpeg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    filtered_image15 = butterworth_highpass_filter(image, d0=15, n=2)
    filtered_image30 = butterworth_highpass_filter(image, d0=30, n=2)
    filtered_image80 = butterworth_highpass_filter(image, d0=80, n=2)
    cv2.imwrite('roses_butterworth_result15.jpg', filtered_image15)
    cv2.imwrite('roses_butterworth_result30.jpg', filtered_image30)
    cv2.imwrite('roses_butterworth_result80.jpg', filtered_image80)


    # Gaussian
    image = cv2.imread("dog.jpeg", cv2.IMREAD_COLOR)
    result_image15 = gauss_high_pass_filter(image, 15)
    result_image30 = gauss_high_pass_filter(image, 30)
    result_image80 = gauss_high_pass_filter(image, 80)
    cv2.imwrite('dog_gauss_result15.jpg', result_image15)
    cv2.imwrite('dog_gauss_result30.jpg', result_image30)
    cv2.imwrite('dog_gauss_result80.jpg', result_image80)

    cv2.waitKey(0)
