# run.py（带参数解析版）
import os
import argparse
import subprocess
from pathlib import Path


def run_training(cuda_devices, config_path, log_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        result = subprocess.run(
            [
                "python", "-m", "model", "train",
                "--verbose",
                "-p", config_path
            ],
            stdout=f,
            stderr=subprocess.STDOUT
        )

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="0", help="device to use")
    # configs/ml1m.json
    # configs/lastfm.json
    # configs/tmall.json
    parser.add_argument("--config", default="configs/tmall.json", help="json config file")
    #  exp/logs/ml1m/CaFBI_ML.log
    #  exp/logs/LastFM/CaFBI_LastFM.log
    #  exp/logs/Tmall/CaFBI_Tmall.log
    parser.add_argument("--log", default="exp/logs/Tmall/CaFBI_Tmall.log", help="log file")

    args = parser.parse_args()

    exit_code = run_training(
        args.cuda,
        args.config,
        args.log
    )

    if exit_code != 0:
        print(f"fail, {exit_code}")
        exit(1)
    else:
        print("train done!")

if __name__ == "__main__":
    main()