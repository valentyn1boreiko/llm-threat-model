import json
import os
import signal
import subprocess
import csv
import wandb
from datetime import datetime

class WandBLogger:
    """WandB logger with support for local logging."""

    def __init__(
        self,
        args,
        project="harmbench-2",
        run_name=None,  # Add parameter for custom run name
        _folder=os.environ.get("WORK_DIR") + "/local_logs/",
    ):
        self.run_data = args if type(args) == dict else vars(args)
        self.project = project
        self.args = args
        print(args)

        # Generate or set a run name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_data["run_id"] = self.run_name  # Add run name to the log data
        self.run_data["run_name"] = self.run_name  # Add run name to the log data

        # Store all logged data for saving to CSV
        self.log_data = []

        # Set W&B to offline mode for local logging
        os.environ["WANDB_MODE"] = "offline"

        self.logger = wandb.init(
            project=project,
            config=args,
            name=self.run_name,  # Assign the run name in WandB
            id=self.run_name
        )
        self.run_id = self.logger.id  # Store the run ID
        self.run_path = self.logger.dir  # Store the run directory path
        self.path_wo_files = self.run_path.rsplit("/", 1)[0]

        self.logging_step = 0

        # Initial log in online mode
        self.initial_log_done = False

        # Set up signal handler for kill signal
        signal.signal(signal.SIGTERM, self._handle_kill_signal)

    def log(self, log_dict, step=None, commit=True):
        # Log using WandB
        self.logger.log(log_dict, step=step, commit=commit)

        # If step is None, use the current logging step
        if step is None:
            step = self.logging_step
        log_dict["step"] = step

        # If it's not the first logging step, update log_dict with previous values
        if self.log_data:
            last_log = self.log_data[-1]  # Get the previous log entry
            # Update missing keys in the new log_dict with values from the previous log entry
            for key, value in last_log.items():
                if key not in log_dict:
                    log_dict[key] = value

        # Append the updated log to the log_data
        self.log_data.append(log_dict)

        # Update the logging step
        self.logging_step += 1

    def _handle_kill_signal(self, signum, frame):
        log_dict = {
            "event": "kill_signal_detected",
            "signal_number": signum,
        }
        self.logger.log(log_dict)

        print(f"Kill signal detected: {signum}")
        self.finish()

    def save_to_json(self):
        """Save the inner dictionary to a JSON file."""
        json_file = os.path.join(self.path_wo_files, f"{self.run_name}.json")
        with open(json_file, "w") as f:
            json.dump(self.run_data, f, indent=4)
        with open(self.log_file, "a") as f:
            f.write(f"Run data saved to {json_file}\n")
        print(f"Run data saved to {json_file}")

    def save_to_pickle(self):
        """Save the inner dictionary to a pickle file."""
        import pickle

        pickle_file = os.path.join(self.path_wo_files, f"{self.run_name}.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(self.run_data, f)
        with open(self.log_file, "a") as f:
            f.write(f"Run data saved to {pickle_file}\n")
        print(f"Run data saved to {pickle_file}")

    def save_to_csv(self):
        """Convert the log data to CSV format and save it, including logging_dict."""
        if not self.log_data:
            print("No log data to save.")
            return

        assert self.args is not None, "The attack has to initialize the logging_dict property"

        # Convert self.args (if it's a Namespace) to a dictionary
        if isinstance(self.args, dict):
            args_dict = self.args
        else:
            args_dict = vars(self.args)  # Convert Namespace to dict


        # Combine logging_dict into each log entry
        all_keys = set(args_dict.keys())  # Start with keys from logging_dict

        # Update the keys based on both log_data and logging_dict
        for log in self.log_data:
            all_keys.update(log.keys())

        csv_file = os.path.join(self.path_wo_files, f"{self.run_name}.csv")

        with open(csv_file, "w", newline='') as f:
            # Create a CSV DictWriter with all the unique keys as fieldnames
            writer = csv.DictWriter(f, fieldnames=all_keys)

            # Write the header row
            writer.writeheader()

            # Write the log entries, merging each log with logging_dict
            for log in self.log_data:
                combined_log = {**args_dict, **log}  # Merge logging_dict and log
                writer.writerow({key: combined_log.get(key, "") for key in all_keys})

        print(f"Run data saved to {csv_file}")

    def finish(self):
        # Finish WandB logging
        self.logger.finish()
        try:
            subprocess.run(["wandb", "sync", self.path_wo_files], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error syncing logs at finish: {e}")
        # Save logs to CSV after finishing
        self.save_to_csv()

