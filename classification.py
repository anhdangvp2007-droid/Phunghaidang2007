# ==============================
# 1. IMPORT THƯ VIỆN
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans, DBSCAN
import scipy.cluster.hierarchy as sch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score

# ==============================
# 2. TẠO DATASET
# ==============================
data = {
    "DoanhThu": [3,6,9,10,12,15,18,20,25,8,14,22],
    "LoiNhuan": [1.1,2.0,3.2,3.4,3.9,5.0,6.2,6.6,8.2,2.5,4.5,7.5],
    "Marketing": [0.5,0.8,1.2,1.5,1.8,2.0,2.5,3.0,3.5,1.0,1.9,3.2],
    "KhachHang": [320,450,700,750,850,1100,1350,1500,1800,600,1000,1650],
    "DanhGia": ["Kém","Kém","Bình thường","Bình thường","Bình thường",
                "Tốt","Tốt","Tốt","Xuất sắc","Kém","Bình thường","Xuất sắc"]
}

df = pd.DataFrame(data)

# ==============================
# 3. PHÂN LOẠI (MULTI-CLASS)
# ==============================

# Encode label
le = LabelEncoder()
df["Label"] = le.fit_transform(df["DanhGia"])

# Feature & Target
X = df[["DoanhThu","LoiNhuan","Marketing","KhachHang"]]
y = df["Label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 4. HUẤN LUYỆN MODEL
# ==============================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted', zero_division=0),
        recall_score(y_test, y_pred, average='weighted', zero_division=0),
        f1_score(y_test, y_pred, average='weighted', zero_division=0)
    ])

# In kết quả
result_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1"])
print("=== KẾT QUẢ PHÂN LOẠI (4 LỚP) ===")
print(result_df)


# ==============================
# 5. PHÂN LOẠI 2 LỚP
# ==============================

df["Label_2"] = df["DanhGia"].replace({
    "Kém": "Kém",
    "Bình thường": "Kém",
    "Tốt": "Tốt",
    "Xuất sắc": "Tốt"
})

le2 = LabelEncoder()
df["Label_2"] = le2.fit_transform(df["Label_2"])

X2 = df[["DoanhThu","LoiNhuan","Marketing","KhachHang"]]
y2 = df["Label_2"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

X2_train = scaler.fit_transform(X2_train)
X2_test = scaler.transform(X2_test)

print("\n=== KẾT QUẢ PHÂN LOẠI (2 LỚP) ===")

for name, model in models.items():
    model.fit(X2_train, y2_train)
    y_pred = model.predict(X2_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y2_test, y_pred))


# ==============================
# 6. PHÂN CỤM (CLUSTERING)
# ==============================

X_cluster = df[["DoanhThu","LoiNhuan","Marketing","KhachHang"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ==============================
# 7. ELBOW METHOD
# ==============================

inertia = []

for k in range(1,6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1,6), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.show()

# ==============================
# 8. KMEANS
# ==============================

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

# ==============================
# 9. HIERARCHICAL CLUSTERING
# ==============================

plt.figure()
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
plt.title("Dendrogram")
plt.show()

# ==============================
# 10. DBSCAN
# ==============================

db = DBSCAN(eps=0.5, min_samples=2)
df["Cluster_DBSCAN"] = db.fit_predict(X_scaled)

# ==============================
# 11. SILHOUETTE SCORE
# ==============================

score_kmeans = silhouette_score(X_scaled, df["Cluster_KMeans"])
print("\nSilhouette Score KMeans:", score_kmeans)

# ==============================
# 12. XEM KẾT QUẢ
# ==============================

print("\n=== DATASET SAU KHI GẮN CLUSTER ===")
print(df)