import mlflow
import os

print("=== MLflow 配置檢查 ===")
print(f"當前工作目錄: {os.getcwd()}")
print(f"MLflow 追蹤 URI: {mlflow.get_tracking_uri()}")
print(f"MLflow 版本: {mlflow.__version__}")

# 檢查環境變數
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
if mlflow_uri:
    print(f"環境變數 MLFLOW_TRACKING_URI: {mlflow_uri}")
else:
    print("沒有設定環境變數 MLFLOW_TRACKING_URI")

# 檢查本地 mlruns 資料夾
if os.path.exists('./mlruns'):
    print("✓ 找到本地 ./mlruns 資料夾")
    print(f"mlruns 內容: {os.listdir('./mlruns')}")
else:
    print("✗ 本地沒有 ./mlruns 資料夾")

# 檢查預設位置
default_locations = [
    './mlruns',
    os.path.expanduser('~/mlruns'),
    '/tmp/mlruns'
]

for location in default_locations:
    if os.path.exists(location):
        print(f"✓ 找到 mlruns 在: {location}")
        try:
            contents = os.listdir(location)
            print(f"  內容: {contents}")
        except:
            print("  無法讀取內容")
