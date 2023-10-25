# Лабораторная работа №1. Сжатие изображений при помощи нейронных сетей

## Requirements
<code>pip install numpy torch torchvision opencv-python opencv-contrib-python Pillow tqdm matplotlib arithmetic-compressor</code>

## Структура проекта
<code>encoder.py</code> — кодировщик на основе предобученного ResNet-18<br>
<code>decoder.py</code> — самописный декодировщик<br>
<code>autoencoder.py</code> — автокодировщик с шумом, соизмеримым по амплитуде с квантованием, между кодированием и декодированием<br>
<code>test_encode.py</code> — тестовый полигон для проверки работоспособности кодировщика с квантованием и сжатием без потерь<br>
<code>test_decode.py</code> — тестовый полигон для проверки работоспособности декодировщика с квантованием и сжатием без потерь<br>
<code>test_autoencode.py</code> — тестовый полигон для проверки работоспособности автокодировщика<br>
<code>train_ae.py</code> — обучение автокодировщика<br>
<code>utils.py</code> — предобработка датасета<br>

Веса для моделей хранятся в папке <code>weights</code>.
Сравнение значений PSNR и графики содержатся в <code>graphics.ipynb</code>

## Примеры запуска
### Encode
<code>python test_encode.py 2 weights/2/ae_epoch3.pt test_images/baboon.png compressed_baboon.txt</code><br><br>
Аргументы:<br>
<ul>
<li>значение коэффициента B (2 или 4);</li>
<li>путь до весов кодировщика;</li>
<li>путь до исходной картинки;</li>
<li>путь до результирующего файла (.txt).</li>
</ul>

### Decode
<code>python test_decode.py 2 weights/2/decode_epoch3.pt compressed_baboon.txt compressed_baboon.png</code><br><br>
Аргументы:<br>
<ul>
<li>значение коэффициента B (2 или 4);</li>
<li>путь до весов декодировщика;</li>
<li>путь до закодированного файла (.txt);</li>
<li>путь до результирующей картинки.</li>
</ul>

### Autoencode
<code>python test_autoencode.py 2 weights/2/ae_epoch3.pt test_images/baboon.png compressed_baboon.png</code><br><br>
Аргументы:<br>
<ul>
<li>значение коэффициента B (2 или 4);</li>
<li>путь до весов автокодировщика;</li>
<li>путь до исходной картинки;</li>
<li>путь до результирующей картинки.</li>
</ul>