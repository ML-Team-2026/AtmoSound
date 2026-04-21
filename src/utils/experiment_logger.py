import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """Log and track hyperparameter experiment results."""

    def __init__(self, name, output_dir="experiment_results"):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.metadata = {
            'name': name,
            'created': datetime.now().isoformat(),
            'num_experiments': 0,
        }

    def log(self, config, metrics):
        """Log a single experiment.

        Args:
            config (dict): Hyperparameter configuration
            metrics (dict): Evaluation metrics
        """
        record = {**config, **metrics}
        self.results.append(record)
        self.metadata['num_experiments'] = len(self.results)

    def to_csv(self, filename=None):
        """Save results to CSV."""
        if filename is None:
            filename = f"{self.name}_results.csv"

        path = self.output_dir / filename
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        return path

    def to_json(self, filename=None):
        """Save results to JSON."""
        if filename is None:
            filename = f"{self.name}_results.json"

        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        return path

    def best(self, metric='test_top3', mode='max'):
        """Get best configuration.

        Args:
            metric (str): Metric column to optimize
            mode (str): 'max' or 'min'

        Returns:
            dict: Best configuration record
        """
        df = pd.DataFrame(self.results)
        if mode == 'max':
            idx = df[metric].idxmax()
        else:
            idx = df[metric].idxmin()
        return df.loc[idx].to_dict()

    def top_k(self, metric='test_top3', k=5, mode='max'):
        """Get top-K configurations.

        Args:
            metric (str): Metric column to optimize
            k (int): Number of results
            mode (str): 'max' or 'min'

        Returns:
            list: List of top-K records
        """
        df = pd.DataFrame(self.results)
        if mode == 'max':
            df = df.nlargest(k, metric)
        else:
            df = df.nsmallest(k, metric)
        return df.to_dict('records')

    def summary(self):
        """Print summary of all experiments."""
        df = pd.DataFrame(self.results)
        print(f"\n{'='*60}")
        print(f"Experiment: {self.name}")
        print(f"Total runs: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print(f"{'='*60}\n")

    def get_dataframe(self):
        """Return results as DataFrame."""
        return pd.DataFrame(self.results)
