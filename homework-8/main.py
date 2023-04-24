import cv2
import numpy as np

def laplasian(image_path):
    # Загрузка исходного изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Вычисление лапласиана в частотной области
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    laplacian = cv2.Laplacian(np.float64(magnitude_spectrum), cv2.CV_64F)
    
    # Градационная коррекция
    alpha = 1.0
    beta = 0.0
    gamma_corrected = np.zeros(laplacian.shape, laplacian.dtype)

    for y in range(laplacian.shape[0]):
        for x in range(laplacian.shape[1]):
            gamma_corrected[y, x] = np.clip(alpha * laplacian[y, x] + beta, 0, 255)

    gamma_corrected = gamma_corrected.astype(np.uint8)
    
    # Получение результата путем вычитания из исходного изображения
    result = cv2.subtract(image, gamma_corrected)

    return result

def high_frequency_emphasis_filter(image, a, b, D0):
    # Проверка входного изображения
    if len(image.shape) > 2:
        print("Входное изображение должно быть в градациях серого")
        return None

    # Получение размеров изображения
    h, w = image.shape

    # Вычисление центра изображения
    center_y, center_x = h // 2, w // 2

    # Применение преобразования Фурье и смещение нулевой частоты к центру
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Создание фильтра H_hp
    H_hp = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
            H_hp[y, x] = 1 - np.exp(-(distance ** 2) / (2 * (D0 ** 2)))

    # Создание передаточной функции H_hfe
    H_hfe = a + b * H_hp

    # Применение фильтра к изображению в частотной области
    filtered_fshift = fshift * H_hfe

    # Обратное смещение нулевой частоты и обратное преобразование Фурье
    f_ishift = np.fft.ifftshift(filtered_fshift)
    filtered_image = np.fft.ifft2(f_ishift)

    # Возврат абсолютного значения и нормализация результата
    filtered_image = np.abs(filtered_image)
    filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image)) * 255

    return filtered_image.astype(np.uint8)

def homomorphic_filter(img, gamma_l, gamma_h, c, d0):
    img = img.astype(np.float32) / 255.0
    rows, cols = img.shape
    img_log = np.log1p(img)

    M, N = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
    padded = cv2.copyMakeBorder(img_log, 0, M - rows, 0, N - cols, cv2.BORDER_CONSTANT, value=0)
    planes = [padded, np.zeros(padded.shape, dtype=np.float32)]
    complex_img = cv2.merge(planes)

    dft_img = cv2.dft(complex_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft_img)

    u, v = np.indices((M, N))
    center_u, center_v = M // 2, N // 2
    d_square = (u - center_u) ** 2 + (v - center_v) ** 2

    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (d_square / (d0 ** 2)))) + gamma_l
    filtered_shift = dft_shift * H[:, :, np.newaxis]

    dft_filtered = np.fft.ifftshift(filtered_shift)
    img_back = cv2.idft(dft_filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    img_exp = np.expm1(img_back)
    img_exp = np.clip(img_exp, 0, 1)

    return (img_exp * 255).astype(np.uint8)

if __name__ == "__main__":

    # Laplacian
    input_image_path = "piter.jpeg"
    output_image_path = "piter_res.jpeg"
    processed_image = laplasian(input_image_path)
    cv2.imwrite(output_image_path, processed_image)

    # High frequencies
    image = cv2.imread("roses.jpeg", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("dog_grey.jpeg", image)
    a = 0.5
    b = 2
    D0 = 30
    filtered_image = high_frequency_emphasis_filter(image, a, b, D0)
    cv2.imwrite("dog_res.jpeg", filtered_image)

    # Homomorphic
    source = 'town'
    source_path = source + '.jpeg'
    img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(source + '_grey.jpeg', img)
    gamma_l = 0.5
    gamma_h = 2.0
    c = 5.0
    d0 = 30.0
    output_image = homomorphic_filter(img, gamma_l, gamma_h, c, d0)
    cv2.imwrite(source + '_res.jpg', output_image)

