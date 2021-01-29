import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    # сразу отсортим фичи и таргет по неубыванию фичей
    ind = np.argsort(feature_vector)
    sorted_feature = feature_vector[ind]
    sorted_target = target_vector[ind]

    # теперь найдём пороги
    # у нас есть отсорченный список фич, мы хотим преобразовать его в вектор арифметических средней соседей
    # вообще говоря это похоже на свёртку: идём по вектору фичей с шагом 1 и применяем свёртку (умножение на вектор (1/2, 1/2))
    thresholds = np.convolve(sorted_feature, np.array([0.5, 0.5]), mode='valid')
    # mode=valid значит что мы применяем свёртку только к "внутренностям"
    # нам это и нужно, ведь пустые множества мы не рассматриваем

    # заметим следующее: если мы делим по определенному порогу (с индексом i),
    # то в левое поддерево попадает i-ая префиксная сумма таргета, а в правое -- i-ая постфиксная
    left_pos = sorted_target.cumsum()[:-1] # последний выкидываем, потому что это случай пустого множества
    # соответственно отрицательные элементы -- это длина массива по которому берётся постфиксная сумма - сама сумма.
    numbers_left = np.arange(1, len(sorted_target))
    left_neg = numbers_left - left_pos

    right_pos = sorted_target.sum() - left_pos
    numbers_right = numbers_left[::-1]
    right_neg = numbers_right - right_pos

    # по сути вот что у нас получилось: numbers_right -- это вектор из |R_r|, numbers_left -- из |R_l| (для каждого пороога)
    # right_pos -- это вектор числа положительных объектов в правом поддереве, right_neg, left_neg, left_pos -- аналогично

    right_impurity = 1 - (right_pos / numbers_right) ** 2 - (right_neg / numbers_right) ** 2
    left_impurity = 1 - (left_pos / numbers_left) ** 2 - (left_neg / numbers_left) ** 2


    ginis = -(right_impurity * numbers_right + left_impurity * numbers_left) / len(sorted_target)

    # проверим что не возникает ситуации когда фичи одинаковые, и мы относим в разные классы объекты с одним значениям
    neighb_equal = (sorted_feature[1:] != sorted_feature[:-1])
    thresholds = thresholds[neighb_equal]
    ginis = ginis[neighb_equal]

    if len(ginis) == 0:
        return thresholds, ginis, None, None

    best_index = ginis.argmax() # argmax при равенстве выдаёт меньший индекс, я проверил
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        # первое условие останова -- все таргеты равны
        # тут и первая ошибка: должно быть == вместо !=
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            # тут была ошибка: рэндж начинался с 1, получается игнорили первый признак
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # ошибка в подсчёте доли. надо поменять числитель и знаменатель
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) # ошибка с индексами тут
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
                # ошибка стоившая мне где то получаса поиска, спасибо хоть stderr открыли
                # np.array от list нужно
            else:
                raise ValueError

            # ещё одна ошибка тут, надо останавливать когда длина вектора 1 (а значит всего 1 сэмпл видимо)
            if len(feature_vector) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini is None:
                continue
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": # буква С была большой. самая гениальная ошибка, спасибо
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # надо индексы ещё поставить было кажись
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        # если конечная вершина то возвращаем класс
        if node['type'] == 'terminal':
            return node['class']

        # иначе делаем разбиение и возвращаем predict_node от поддерева
        feature_split = node["feature_split"]
        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif self._feature_types[feature_split] == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            raise ValueError


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep):
        return {'feature_types': self._feature_types}
