import os
import argparse
import subprocess
from pathlib import Path


def run_evaluation(cuda_devices, config_path, log_path, best_epoch):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        result = subprocess.run(
            [
                "python", "-m", "model", "eval",
                "--verbose",
                "-p", config_path,
                "--best_epoch", str(best_epoch)
            ],
            stdout=f,
            stderr=subprocess.STDOUT
        )

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="device to use")
    # ml1m.json / lastfm.json / tmall.json
    parser.add_argument("--config", default="configs/tmall.json", help="json config file")
    # exp/logs/ml1m/CaFBI_ML_eval.log   exp/logs/LastFM/CaFBI_LastFM_eval.log   exp/logs/Tmall/CaFBI_Tmall_eval.log
    parser.add_argument("--log", default="exp/logs/Tmall/CaFBI_Tmall_eval.log", help="log file")
    parser.add_argument("--best_epoch", default=30, type=int, help="the value of best epoch")

    args = parser.parse_args()

    exit_code = run_evaluation(
        args.cuda,
        args.config,
        args.log,
        args.best_epoch
    )

    if exit_code != 0:
        print(f"fail, {exit_code}")
        exit(1)
    else:
        print("evaluation done!")


if __name__ == "__main__":
    main()
