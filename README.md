# ДЗ-13. Компьютерное зрение

[Описание домашнего задания](https://otusmetodist.yonote.ru/share/2cb07ace-75b1-48fe-b6fd-259b39f6025a/doc/13-kompyuternoe-zrenie-YWRgYevDd5)

## Сборка в Docker

Для сборки проекта требуется только Docker. Компилятор, CMake, Eigen и
GoogleTest устанавливаются внутри образа.

Сборка образа из корня проекта:

```bash
docker build -t fashion-mnist .
```

Для полностью чистой сборки с обновлением базового образа:

```bash
docker build --pull --no-cache -t fashion-mnist .
```
