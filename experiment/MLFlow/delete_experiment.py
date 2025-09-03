from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()

# 列出所有實驗
for exp in client.search_experiments(filter_string="", view_type=ViewType.ALL):
    print(exp.experiment_id, exp.name, exp.lifecycle_stage)

# 刪除指定名稱的實驗
exp_name = "CNN_MNIST_Demo"
exp = client.get_experiment_by_name(exp_name)
if exp:
    client.delete_experiment(exp.experiment_id)
    print(f"已刪除實驗: {exp_name}")
else:
    print(f"找不到實驗: {exp_name}")
