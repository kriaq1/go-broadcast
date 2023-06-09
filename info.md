# Действия
&nbsp;
## Choose source - выбрать камеру/видео/стрим в списке. Нельзя в списке выбирать камеру, которая уже используется.  
&nbsp;
## add source - добавить в список видео по указанному пути  
&nbsp;
## add url - добавить в список url на видео
&nbsp;
## Save SGF - выбрать путь для сохранения в sgf текущей позиции, которая отображается на доске
&nbsp;
## Каждые 10 секунд картинка из видео сохраняется в папке temp, при закрытии программы или изменения камеры/видео/стрима все сохраненные картинки удаляются. 
&nbsp;
## Можно проматывать видео, состояние доски не проматывается, распознавание не прекращается. Чтобы продолжить показывать видео, нужно уставновить ползунок максимально вправо.
&nbsp;
## Цифры на камнях - их порядковый номер. Камни, помеченные красным - будущие ходы, которые еще не добавлены в sgf на случай неправильно распознанных камней.
&nbsp;
## После каждого найденного хода обновляется файл sgf.sgf в текущей папке.
&nbsp;
## start recognition - начать распознавание. В подпроцесс передается выбранное видео, компонент распознавания начинает непрерывно читать картинку, распознавать состояние доски и добавлять в очередь полученное состояние доски. Другой компонент читает из очереди эти состояния и определяет последовательность ходов. Если на доске уже есть камни, они расставятся случайно.
&nbsp;
## stop recognition - остановить распознавание. Камни, помеченные красным добавляются в sgf. Компонент, определяющий последовательность ходов не перестает работать. После stop recognition можно нажать start recognition и распознавание продолжит работу.
&nbsp;
## set current state - установить текущее состояние доски на картине, последовательность ходов расставить случайно. Компонент, определяющий последовательность ходов очищается и заполняется случайными ходами, sgf очищается и заполняется случайными ходами. Нужно нажимать при распознавании новой партии.
&nbsp;
## use previous points - использовать/не использовать текущие найденные координаты доски, отключить/включить постоянный поиск доски. Поиск доски занимает много времени, из-за чего увеличивается период между картинками для распознавания и таким образом ухудшается точность определения последовательности ходов, так как между двумя хорошими картинками могут быть несколько ходов. Использование предыдущих найденных координат ускоряет работу распознавания, но при сильном сдвиге доски не находит камни, не попадающие в область доски. Если доску сдвинули, можно отключить и включить заново, чтобы обновить координаты доски
&nbsp;
## set points - после первого нажатия считывает нажатия левой кнопки мышки на видео, изменяет ближайшую координату угла доски. После второго нажатия отправляет новые координаты в компонент распознавания и устанавливает 'use previous points'
&nbsp;


# Настройка параметров распознавания (Settings -> Parameters -> Recognition):
&nbsp;
## При нажатии apply распознаванию отправляются введенные параметры, при нажатии cancel отменяются последние изменения.
&nbsp;
## Распознавание читает видео, в очередь закидывает состояние доски, вероятности, качество сегментации, время, найденные координаты.
&nbsp;
## conf - пороговая вероятность камней на доске. Если камни не находятся, может помочь уменьшение параметра. Если камни стоят на неправильных позициях, может помочь увеличение параметра.
&nbsp;
## iou - коэффициент, при котором удаляются рядом стоящие координаты камней. Если два камня сливаются в один, может помочь увеличение параметра. Если один камень отображается как два, может помочь уменьшение параметра.
&nbsp;
## min_distance - минимальное расстояние на вырезанной картинке между узлами сетки.
&nbsp;
## max_distance - максимальное расстояние на вырезанной картинке между узлами сетки. По этим расстояниям строится граф между камнями и узлами сетки и находится наиболее подходящая часть 19х19 в нём
&nbsp;
## quality_coefficient - после нахождения координат доски проверяется качество найденных координат и если это качество меньше чем предыдущее качество умноженное на этот параметр, то эти координаты пропускаются. Если на видео очень часто появляются руки над доской, то при увеличении этого параметра координаты доски будут более постоянными
&nbsp;
## search_period - период поиска координат доски. Если search_period = 4, то распознавание находит координаты доски, 4 раза считывает картинку и находит камни на вырезанной с помощью координат картинке, далее заново запускает поиск координат и так далее. Увеличение этого параметра ускорит работу распознавания, но повышает вероятность долгого использования плохо найденных координат. При увеличении search_period полезно увеличивать quality_coefficient.
&nbsp;
## seg_threshold - пороговая вероятность при поиске координат доски. Если доска не находится, может помочь сильное уменьшение параметра.
&nbsp;
# Настройка параметров компонента для определения последовательности ходов (Settings -> Parameters -> Logging):
&nbsp;
## Компонент получает состояния доски от распознавания. Определяются хорошо распознанные состояния, между ними смотрят на плохо распознанные состояния и пытаются восстановить последовательность ходов.  
&nbsp;
## appearance_count - количество картинок, на которых распознавание нашло определенный камень, чтобы при восстановлении ходов между состояниями считать появление этого камня не выбросом (руки иногда могут распознаться как камни). Уменьшение параметра повышает точность восстановления последовательности, увеличивает чувствительность к выбросам
&nbsp;
## delay - задержка для хорошо распознанных состояний. Помеченные красным камни - камни из этой задержки. Если в новом хорошо распознанном состоянии нет камня, который есть в состояниях из задержки и этот камень не удалялся по правилам, то из состояний в задержке удалится этот камень. Если установить delay=0, то не будет помеченных красным камней - они сразу добавятся в sgf. 
&nbsp;
## valid_zeros_count - максимальное количество камней/пустых клеток, в распознавании которых не уверены для определения хорошего состояния. Те позиции, в которых не уверены, заполняются предыдущим хорошим состоянием и получается новое хорошее состояние.
&nbsp;
## valid_thresh - пороговая вероятность для определения уверенности в камне
&nbsp;
## valid_time - промежуток времени, на котором считается уверенность камней для нахождения хорошего состояния. Если над доской часто появляются руки или долго появляются новые ходы, может помочь уменьшение этого параметра.
