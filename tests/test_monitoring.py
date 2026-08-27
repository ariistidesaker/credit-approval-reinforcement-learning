import os
import pytest
import pandas as pd
from src.monitoring import TrainingLogger


def test_training_logger_operations(tmp_path):
    log_dir = str(tmp_path / "runs")
    reports_dir = str(tmp_path / "reports")

    logger = TrainingLogger(log_dir=log_dir, experiment_name="test_exp", use_tensorboard=True)

    # Log 5 test episodes
    for ep in range(1, 6):
        logger.log_episode(
            episode=ep,
            ep_reward=10.0 * ep,
            total_profit=100_000.0 * ep,
            approval_rate=0.7,
            default_rate=0.2,
            volume_lent=500_000.0,
            roi=0.2,
            policy_loss=0.1 / ep,
            value_loss=0.5 / ep,
            entropy=0.6,
            approx_kl=0.01,
            eval_metrics={"eval_mean_profit": 110_000.0 * ep, "eval_mean_reward": 11.0 * ep} if ep % 2 == 0 else None,
        )

    assert len(logger.history) == 5
    assert logger.history[0]["episode"] == 1
    assert "rolling_mean_reward" in logger.history[0]

    # Test plot generation
    plot_path = os.path.join(reports_dir, "test_curves.png")
    logger.plot_learning_curves(save_path=plot_path)
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 1000

    # Test CSV export
    csv_path = os.path.join(reports_dir, "test_history.csv")
    exported_path = logger.export_csv(save_path=csv_path)
    assert os.path.exists(exported_path)

    df = pd.read_csv(exported_path)
    assert len(df) == 5
    assert "total_profit" in df.columns

    logger.close()
