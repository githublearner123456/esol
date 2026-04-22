# 导入库
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, Lipinski
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv(r"C:\Users\16544\Downloads\esol.csv")

# ================== 1. 提取额外分子描述符（加入 Minimum Degree）==================
extra_cols = [
    'Minimum Degree',
    'Molecular Weight',
    'Number of H-Bond Donors',
    'Number of Rings',
    'Number of Rotatable Bonds',
    'Polar Surface Area'
]
X_extra_raw = data[extra_cols].values.astype(np.float32)

# 处理缺失值（用中位数填充）
imputer = SimpleImputer(strategy='median')
X_extra = imputer.fit_transform(X_extra_raw)

# ================== 2. Morgan 指纹生成 ==================
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(2048)
        fp = morgan_gen.GetFingerprint(mol)
        return np.array(fp)
    except:
        return np.zeros(2048)

print("正在提取 Morgan 指纹...")
X_fp = np.array([smiles_to_fp(smi) for smi in data["smiles"]])
print("指纹提取完成")

# 标签：使用实测 log 溶解度
y = data["measured log solubility in mols per litre"].values.reshape(-1, 1)

# ================== 3. 划分数据集 ==================
X_temp_fp, X_test_fp, X_temp_extra, X_test_extra, y_temp, y_test = train_test_split(
    X_fp, X_extra, y, test_size=0.1, random_state=42
)
X_train_fp, X_val_fp, X_train_extra, X_val_extra, y_train, y_val = train_test_split(
    X_temp_fp, X_temp_extra, y_temp, test_size=0.1111, random_state=42
)

print(f"数据集划分 -> 训练集: {len(X_train_fp)}, 验证集: {len(X_val_fp)}, 测试集: {len(X_test_fp)}")

# ================== 4. 标准化额外特征 ==================
scaler = StandardScaler()
X_train_extra_scaled = scaler.fit_transform(X_train_extra)
X_val_extra_scaled   = scaler.transform(X_val_extra)
X_test_extra_scaled  = scaler.transform(X_test_extra)

# ================== 5. 拼接特征 ==================
X_train = np.hstack([X_train_fp, X_train_extra_scaled])
X_val   = np.hstack([X_val_fp, X_val_extra_scaled])
X_test  = np.hstack([X_test_fp, X_test_extra_scaled])

# ================== 6. 转为张量 ==================
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
y_val_t   = torch.tensor(y_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

# ================== 7. 定义模型（输入维度 = 2048 + 6）==================
input_dim = X_train.shape[1]   # 2048 + 6 = 2054
model = nn.Sequential(
    nn.Linear(input_dim, 16),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(16, 1)
)

loss_fn = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)

# 记录列表
train_losses = []
train_rmse_list = []
train_r2_list = []
val_rmse_list = []
val_r2_list = []

# 早停参数（基于验证集 R²，patience=10）
patience = 10
best_val_r2 = -float('inf')
best_model_state = None
epochs_no_improve = 0
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(X_train_t)
    train_losses.append(train_loss)
    
    # 训练集 RMSE 和 R²
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).cpu().numpy()
        train_true = y_train_t.cpu().numpy()
        train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
        train_r2 = r2_score(train_true, train_pred)
    train_rmse_list.append(train_rmse)
    train_r2_list.append(train_r2)
    
    # 验证集 RMSE 和 R²
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
        val_true = y_val_t.cpu().numpy()
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
        val_r2 = r2_score(val_true, val_pred)
    val_rmse_list.append(val_rmse)
    val_r2_list.append(val_r2)
    
    print(f"Epoch {epoch+1:3d}/{num_epochs} | "
          f"Train Loss (MSE): {train_loss:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f} | "
          f"Val RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    
    # 早停（基于验证集 R²）
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_model_state = model.state_dict().copy()
        epochs_no_improve = 0
        print(f"  ^ 验证集 R² 改善 ({best_val_r2:.4f})，保存模型")
    else:
        epochs_no_improve += 1
        print(f"  ! 验证集 R² 未改善，连续 {epochs_no_improve}/{patience} 轮")
        if epochs_no_improve >= patience:
            print(f"\n早停触发！验证集 R² 连续 {patience} 轮未改善，停止训练。")
            break

# 加载最佳模型
model.load_state_dict(best_model_state)
print(f"\n最佳验证集 R²: {best_val_r2:.4f} (在第 {val_r2_list.index(best_val_r2)+1} 轮)")

# ================== 8. 测试集评估 ==================
model.eval()
with torch.no_grad():
    test_pred = model(X_test_t).cpu().numpy()
    y_test_np = y_test_t.cpu().numpy()
    test_rmse = np.sqrt(mean_squared_error(y_test_np, test_pred))
    test_r2 = r2_score(y_test_np, test_pred)
    print("\n最终测试集评估结果：")
    print(f"  RMSE : {test_rmse:.4f}")
    print(f"  R²   : {test_r2:.4f}")

# ================== 9. 画图 ==================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_rmse_list, label='Train RMSE', linewidth=2, marker='o', markersize=3)
plt.plot(val_rmse_list, label='Val RMSE', linewidth=2, marker='s', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE on Training and Validation Sets')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_r2_list, label='Train R²', linewidth=2, marker='o', markersize=3)
plt.plot(val_r2_list, label='Val R²', linewidth=2, marker='s', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.title('R² on Training and Validation Sets')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 过拟合判断
best_val_rmse = min(val_rmse_list)
if best_val_rmse > train_rmse_list[-1] * 1.2:
    print("⚠️ 警告：可能存在过拟合！最佳验证RMSE比最终训练RMSE高20%以上")
elif best_val_rmse > train_rmse_list[-1] * 1.1:
    print("⚠️ 轻微过拟合")
else:
    print("✅ 没有明显过拟合")

# ================== 10. 预测新分子（包含 Minimum Degree 计算）==================
def compute_extra_descriptors(mol):
    """从 RDKit 分子对象计算 6 个描述符（包括 Minimum Degree）"""
    min_degree = min([atom.GetDegree() for atom in mol.GetAtoms()]) if mol.GetNumAtoms() > 0 else 0
    mol_weight = Descriptors.MolWt(mol)
    hbd = Lipinski.NumHDonors(mol)
    n_rings = Descriptors.RingCount(mol)
    n_rot = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    return np.array([min_degree, mol_weight, hbd, n_rings, n_rot, tpsa], dtype=np.float32)

def predict_solubility(smiles):
    fp = smiles_to_fp(smiles).reshape(1, -1)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的 SMILES 字符串")
    extra_raw = compute_extra_descriptors(mol).reshape(1, -1)
    extra_filled = imputer.transform(extra_raw)
    extra_scaled = scaler.transform(extra_filled)
    X = np.hstack([fp, extra_scaled])
    X_t = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(X_t).item()
    return pred

# 示例
test_smiles = "Clc1ccc(c(Cl)c1Cl)c2c(Cl)cc(Cl)c(Cl)c2Cl "
pred_val = predict_solubility(test_smiles)
print(f"\n预测2,3,3',4,4'6-PCB 溶解度: {pred_val:.4f} (真实值-7.66 )")
