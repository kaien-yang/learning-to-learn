import torch
import numpy as np

class SinusoidDataset():
    def get_data(self, batch_size, num_shots_support, num_shots_query, num_tasks):
        """Generates sinusoid regression data.

        Args:
            batch_size (int): Number of tasks per outer-loop update.
            num_support (int): Number of support examples in a task.
            num_query (int): Number of query examples in a task.
            num_tasks (int): Total number of tasks. Must be divisible
                by batch_size.

        Returns:
            list[list[list[Tensor]]]: Three nested lists, with lengths
                `batch_size`, `num_tasks // batch_size`, and 4, from
                outermost to innermost, respectively.
                Entries of innermost list are:
                    support x values of shape (`num_support`, 1);
                    support y values of shape (`num_support`, 1);
                    query x values of shape (`num_support`, 1); and
                    query y values of shape (`num_support`, 1).
        """

        data = []
        for train_iter in range(num_tasks//batch_size):
            batch = []
            for task in range(batch_size):
                amplitude = np.random.uniform(0.1, 5)
                phase = np.random.uniform(0, np.pi)
                x_support = np.random.uniform(-5, 5, num_shots_support)
                y_support = amplitude * np.sin(phase + x_support)
                x_query = np.random.uniform(-5, 5, num_shots_query)
                y_query = amplitude * np.sin(phase + x_query)
                batch.append([
                    torch.tensor(x_support, dtype=torch.float)[:, None],
                    torch.tensor(y_support, dtype=torch.float)[:, None],
                    torch.tensor(x_query, dtype=torch.float)[:, None],
                    torch.tensor(y_query, dtype=torch.float)[:, None],
                ])
            data.append(batch)
        return data