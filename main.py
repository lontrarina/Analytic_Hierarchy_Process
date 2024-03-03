import numpy as np
import pandas as pd
from colorama import Fore, Style

def matrix_vector_multiplication(matrix_A, vector_x):
    # Перевірка розмірності матриці та вектора
    if matrix_A.shape[1] != len(vector_x):
        return "Розмірності матриці та вектора не підходять для множення."

    result = np.dot(matrix_A, vector_x)
    return result

def calculate_v(matrix_A, k):
    # Обчислення добутку елементів кожного рядка та взяття кореня ступеня k
    v = np.power(np.prod(matrix_A, axis=1), 1/k)
    return v


def calculate_w(v):
    sum_v = np.sum(v)
    w = v/ sum_v
    return w

def read_matrix_from_file(file_path):
    matrix_A = np.loadtxt(file_path)
    return matrix_A



def find_local_priorities(matrix_A, k):
    print("Обчислення пріоритетів як середніх геометричних рядків МПП ")
    v = calculate_v(matrix_A, k)
    print("Вектор v: ", v)

    w = calculate_w(v)
    print(f"Вектор W: {w}")
    return w

def estimate_consistency(matrix_A, CRITERIA):
    # Кількість ітерацій
    k=CRITERIA
    num_iterations = 10
    CIS = [0, 0, 0.52, 0.89, 1.11, 1.25, 1.35, 1.4, 1.45, 1.49]

    x_m = np.ones((matrix_A.shape[0], 1))

    print('Метод простої векторної ітерації:')
    for i in range(num_iterations):
        x_m1 = matrix_vector_multiplication(matrix_A, x_m[:, i])
        x_m = np.concatenate((x_m, x_m1.reshape(-1, 1)), axis=1)

    # Перетворення матриці x_m на об'єкт DataFrame бібліотеки Pandas
    df = pd.DataFrame(x_m)
    print(df)

    print("Побудуємо послідовність x[m+1]/x[m] для i=1:")
    ratio = x_m[0, 1:] / x_m[0, :-1]
    ratio_df = pd.DataFrame(ratio)
    print(ratio_df)

    e = 0.001  # Задана точність
    lambda_max = None  # Ініціалізуємо lambda_max
    for i in range(10):
        difference = abs(ratio[i] - ratio[i - 1])
        if difference < e:
            lambda_max = ratio[i]
            print(f"lambda_max = {lambda_max}, точність e={e}")
            break

    # Якщо lambda_max не обрано, коли різниця < e, виводимо повідомлення про це
    if lambda_max is None:
        print(f"lambda_max не обрано, оскільки різниця не стала меншою за точність e={e}")

    CI = (lambda_max - k) / (k - 1)
    print(f"Iндекс узгодженості CI={CI}")

    CR = CI / CIS[k - 1]
    print(f"Відношення узгодженості CR={CR}")

    consistency= False
    if(CI<=0.1)and (CR<=0.1):
        if(k==3):
            if (CR<=0.05):
                consistency=True
            else: consistency= False
        elif (k==4):
            if (CR<=0.08):
                consistency = True
            else: consistency = False
        else:
            consistency = True
    else:
        print(Fore.RED + "Матриця НЕдостатнього ступеня неузгодженості для використання.------------------" + Style.RESET_ALL)

    return consistency



def read_matrix_from_file(file_path, start_line, end_line):
    # Читаємо дані з файлу між визначеними рядками
    with open(file_path, 'r') as file:
        lines = file.readlines()[start_line - 1:end_line]

    # Створюємо список списків, де кожний список - це рядок матриці
    matrix_data = [list(map(float, line.split())) for line in lines]

    # Перетворюємо список списків у масив NumPy
    matrix = np.array(matrix_data)

    return matrix

file_path = "Варіант №10 умова.txt"
EXPERTS=4
CRITERIA=4
ALTERNATIVES=6

MPP_Experts=read_matrix_from_file(file_path, 3, 6)

MPP_Criterias_by_Expert=np.array([
    read_matrix_from_file(file_path, 8, 11), #expert1
    read_matrix_from_file(file_path, 41, 44), #expert2
    read_matrix_from_file(file_path, 74, 77), #expert3
    read_matrix_from_file(file_path, 107, 110), #expert4
])

MPP_Alternatives_by_Criteria=np.array([
    [   #expert1
        read_matrix_from_file(file_path, 13, 18),
        read_matrix_from_file(file_path, 20, 25),
        read_matrix_from_file(file_path, 27, 32),
        read_matrix_from_file(file_path, 34, 39)
    ],
    [  # expert2
        read_matrix_from_file(file_path, 46, 51),
        read_matrix_from_file(file_path, 53, 58),
        read_matrix_from_file(file_path, 60, 65),
        read_matrix_from_file(file_path, 67, 72)
    ],
    [  # expert3
        read_matrix_from_file(file_path, 79, 84),
        read_matrix_from_file(file_path, 86, 91),
        read_matrix_from_file(file_path, 93, 98),
        read_matrix_from_file(file_path, 100, 105)
    ],
    [  # expert4
        read_matrix_from_file(file_path, 112, 117),
        read_matrix_from_file(file_path, 119, 124),
        read_matrix_from_file(file_path, 126, 131),
        read_matrix_from_file(file_path, 133, 138)
    ],

])




print("\n\n                            ВИЗНАЧЕННЯ ЛОКАЛЬНИХ ВАГОВИХ КОЕФІЦІЄНТІВ ЕКСПЕРТІВ")
k=np.zeros(shape=(EXPERTS))
is_consistency_Exp = estimate_consistency(MPP_Experts, EXPERTS)

if (is_consistency_Exp == False):
    print(Fore.RED + "Матриця НЕдостатнього ступеня неузгодженості для використання.------------------" + Style.RESET_ALL)
else:
    print(Fore.GREEN + "Матриця ДОСТАТНЬОГО ступеня неузгодженості для використання.-----------------" + Style.RESET_ALL)
    k= find_local_priorities(MPP_Experts, EXPERTS)
    print("\n\n")
print("\n\n")

print("\n\n                              ВИЗНАЧЕННЯ ЛОКАЛЬНИХ ВАГОВИХ КОЕФІЦІЄНТІВ КРИТЕРІЇВ")
w=np.zeros(shape=(EXPERTS, CRITERIA))
for expert_idx, matrix in enumerate(MPP_Criterias_by_Expert):

    print(f"Коефіцієнти критеріїв для експерта {expert_idx+1}:")
    is_consistency_CrbEx = estimate_consistency(matrix, CRITERIA)
    if is_consistency_CrbEx == False:
        print(Fore.RED + "Матриця НЕдостатнього ступеня неузгодженості для використання.------------------" + Style.RESET_ALL)

    else:
        print(Fore.GREEN + "Матриця ДОСТАТНЬОГО ступеня неузгодженості для використання.-----------------" + Style.RESET_ALL)
        w_i = find_local_priorities(matrix, CRITERIA)
        w[expert_idx] = w_i
        print("\n\n")

print("\n\n")


print("\n\n                                  ВИЗНАЧЕННЯ ЛОКАЛЬНИХ ВАГОВИХ КОЕФІЦІЄНТІВ АЛЬТЕРНАТИВ")
p = np.zeros(shape=(EXPERTS, CRITERIA, ALTERNATIVES))
for i, expert_matrices in enumerate(MPP_Alternatives_by_Criteria):
    expert_number = i + 1
    for j, criteria_matrix in enumerate(expert_matrices):
        criteria_number = j + 1
        print(f"Коефіцієнти альтернатив для експерта {expert_number} та критерія {criteria_number}:")
        is_consistency_AltbCr = estimate_consistency(criteria_matrix, ALTERNATIVES)

        if (is_consistency_AltbCr == False):
            print(Fore.RED + "Матриця НЕдостатнього ступеня неузгодженості для використання.------------------" + Style.RESET_ALL)
        else:
            print( Fore.GREEN + "Матриця ДОСТАТНЬОГО ступеня неузгодженості для використання.-----------------" + Style.RESET_ALL)
            p[i, j] = find_local_priorities(criteria_matrix, ALTERNATIVES)
            print("\n\n")



print("\n\n                              ЛОКАЛЬНІ ВАГОВІ КОЕФІЦІЄНТИ КРИТЕРІЇВ")
for expert_idx, matrix in enumerate(MPP_Criterias_by_Expert):
    print(f"Коефіцієнти критеріїв для експерта {expert_idx+1}:")
    print(w[expert_idx])
print("\n\n")


print("\n\n                                  ЛОКАЛЬНІ ВАГОВІ КОЕФІЦІЄНТИ АЛЬТЕРНАТИВ")
for i, expert_matrices in enumerate(MPP_Alternatives_by_Criteria):
    expert_number = i + 1
    print('\n')
    for j, criteria_matrix in enumerate(expert_matrices):
        criteria_number = j + 1
        print(f"Коефіцієнти альтернатив для експерта {expert_number} та критерія {criteria_number}:")
        print(p[i, j])




print("\n\n                           ВИЗНАЧЕННЯ ГЛОБАЛЬНИХ ВАГОВИХ КОЕФІЦІЄНТІВ АЛЬТЕРНАТИВ")
global_weights=np.array([
    np.sum([
        k[s]*np.sum([p[s][j][i]*w[s][j] for j in range(CRITERIA)])
        for s in range(EXPERTS)
    ])
    for i in range(ALTERNATIVES)
])
print(global_weights)
print(f"Сума глобaльних вагових коефіцієнтів= {sum(global_weights)}")