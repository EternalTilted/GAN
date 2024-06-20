# GAN

### Финальная версия

Эта версия была обучена на всем том же самом датасете. Было выявлено нескоько ошибок при проектировке, а именно малое скрытое пространство, из-за чего нейронная сеть не способна на детальную прорисовку объектов.
Также можно отметить, что дискриминатор и генератор имеют слишком большую глубину.

Замечания по обучению: для большей скорости обучения можно переписать формулу потерь, так как выходит, что на ранних этапах обучения график имеет логарифмический вид, но спустя некоторое время он становится линейным.

Так как исправление и переобучение нейронной сети займёт слишком много времени, на данный момент этот проект сворачивается.

### График потерь
![loss_graphic](https://github.com/EternalTilted/GAN/assets/94389581/f19f4999-e049-44a8-82ca-77a65c16cb30)


### Примеры генераций
![generate_pic_7](https://github.com/EternalTilted/GAN/assets/94389581/64227513-7144-4505-bd62-919fe34e4e4e)

![generate_pic_10](https://github.com/EternalTilted/GAN/assets/94389581/0d7b6f81-ea74-492c-aaf0-ddcfcc5b9780)

![generate_pic_13](https://github.com/EternalTilted/GAN/assets/94389581/334ff87b-eb5d-44d0-b243-0ebad9e19a5e)

![generate_pic_13-](https://github.com/EternalTilted/GAN/assets/94389581/8b6a5240-312b-456e-8a5d-27b915f4ac27)

![generate_pic_28](https://github.com/EternalTilted/GAN/assets/94389581/c2e717ae-904b-4a5f-bb9e-b3628591cbc7)

![generate_pic_40](https://github.com/EternalTilted/GAN/assets/94389581/233b3a5b-b51e-4c57-9f8a-ea9c365f2f54)

![generate_pic_89](https://github.com/EternalTilted/GAN/assets/94389581/585225d1-732a-4179-af19-1e950e06c53f)

![generate_pic_73](https://github.com/EternalTilted/GAN/assets/94389581/fc6bf939-dd54-49e4-be52-d95f782f54ca)
