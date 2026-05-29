from tools.pose_trainer import PoseTrainer

trainer = PoseTrainer(
    data_dir="data/datasets"
)

trainer.train()
trainer.export("model/pose_model.pkl")