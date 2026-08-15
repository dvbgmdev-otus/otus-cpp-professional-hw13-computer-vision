# ДЗ-13. Компьютерное зрение

[Описание домашнего задания](https://otusmetodist.yonote.ru/share/2cb07ace-75b1-48fe-b6fd-259b39f6025a/doc/13-kompyuternoe-zrenie-YWRgYevDd5)

Программа выполняет инференс многослойного перцептрона на проверочной выборке
Fashion MNIST и выводит долю правильно классифицированных изображений.

## Входные данные

Файл `data/test.csv` содержит строки из правильного класса и 784 пикселей
изображения 28x28. Каталог `model` содержит матрицы весов `w1.txt` и `w2.txt`.

## Сборка в Docker

Для сборки требуется только Docker. Компилятор, CMake, Eigen и GoogleTest
устанавливаются внутри образа.

```bash
docker build -t fashion-mnist .
```

Во время сборки автоматически компилируются и запускаются тесты. Для полностью
чистой сборки с обновлением базового образа:

```bash
docker build --pull --no-cache -t fashion-mnist .
```

## Запуск

```bash
docker run --rm fashion-mnist
```

Ожидаемый результат:

```text
0.884889
```

Программу также можно вызвать внутри образа с явными путями:

```bash
docker run --rm fashion-mnist \
    ./build/fashion_mnist data/test.csv model
```

## Тесты

Повторный запуск тестов без пересборки образа:

```bash
docker run --rm --workdir /workspace/build \
    fashion-mnist ctest --output-on-failure
```
