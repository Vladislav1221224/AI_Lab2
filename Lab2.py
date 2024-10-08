import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Генерація випадкового набору даних у діапазоні 1000 значень
np.random.seed(42)
X = np.random.rand(1000, 1) * 1000  # 1000 випадкових значень від 0 до 1000
y = X + np.random.randn(1000, 1) * 50  # Невеликий шум до даних

# 2. Нормалізація значень
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Розділити записи на навчальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Навчити KNN-регресор з різними значеннями K
k_values = [3, 5, 10]
knn_models = {}
errors = {}

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_models[k] = knn
    y_pred = knn.predict(X_test)
    errors[k] = mean_squared_error(y_test, y_pred)

# 5. Вибір K з найкращими показниками якості регресії
best_k = min(errors, key=errors.get)
print(f"Найкраще значення K: {best_k} з помилкою {errors[best_k]}")

# 6. Візуалізація: Тренувальні точки, Тестові точки і Прогнози

# Графік 1: Тренувальні точки
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='green', label='Тренувальні точки', alpha=0.6)
plt.title('Тренувальні дані')
plt.xlabel('X (нормалізовані значення)')
plt.ylabel('y')
plt.legend()
plt.show()

# Графік 2: Тестові точки
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Тестові точки', alpha=0.6)
plt.title('Тестові дані')
plt.xlabel('X (нормалізовані значення)')
plt.ylabel('y')
plt.legend()
plt.show()

# Графік 3: Прогнози для різних значень K
plt.figure(figsize=(10, 6))
colors = ['red', 'purple', 'orange']  # кольори для різних значень K
for i, k in enumerate(k_values):
    y_pred = knn_models[k].predict(X_test)
    plt.scatter(X_test, y_pred, color=colors[i], label=f'Прогнози K={k}', alpha=0.6)

plt.title('Прогнози KNN-регресії для різних K')
plt.xlabel('X (нормалізовані значення)')
plt.ylabel('y')
plt.legend()
plt.show()

# 6. Візуалізація отриманих рішень
plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='blue', label='Тестові дані', alpha=0.4)
for k in k_values:
    y_pred = knn_models[k].predict(X_test)
    plt.scatter(X_test, y_pred, label=f'Прогнози K={k}', alpha=0.6)

plt.title('Прогнози KNN-регресії з різними значеннями K')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
