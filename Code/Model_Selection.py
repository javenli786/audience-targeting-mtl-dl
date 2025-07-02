import mlflow
from mlflow.tracking import MlflowClient

def Transition_to_Production(model_name="DonationMaximization_MTL_Model"):

    client = MlflowClient()
    
    # 获取所有模型版本
    versions = client.search_model_versions(f"name='{model_name}'")
    
    # 找出验证损失最低的版本
    best_version = None
    best_loss = float('inf')
    
    for v in versions:
        if v.run_id:
            run = client.get_run(v.run_id)
            val_loss = run.data.metrics.get("best_val_loss", float('inf'))
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_version = v.version
    
    if best_version:
        # 设置最佳版本为生产环境
        client.transition_model_version_stage(
            name=model_name,
            version=best_version,
            stage="Production"
        )
        
        # 归档其他版本
        for v in versions:
            if v.version != best_version and v.current_stage != "Archived":
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived"
                )
        
        print(f"已将版本 {best_version} (验证损失: {best_loss:.6f}) 设为生产环境，归档其他版本")
        return best_version
    else:
        print(f"未找到可用的模型版本")
        return None