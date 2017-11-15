import h5py
import numpy as np

from utils import BackgroundGenerator


def get_data_h5py(self, fname="data-experience.0.hdf5"):
    # Open file handle, but don't load contents into memory
    h5f = h5py.File(fname, "r")
    entries = len(h5f['grids'])

    split_perc = 0.9  # Percentage of data to train on
    split = int(entries * split_perc) // self.batch_size
    end = entries // self.batch_size

    def data_gen(start, stop):
        for i in range(start, stop):
            batch = slice(i * self.batch_size, (i + 1) * self.batch_size)
            # Load batch data into memory and prep it
            grids, cells, actions, rewards, next_grids, next_cells = \
                self.prep_data(
                    h5f['grids'][batch][:],
                    h5f['cells'][batch][:],
                    h5f['chs'][batch][:],
                    h5f['rewards'][batch][:],
                    h5f['next_grids'][batch][:],
                    h5f['next_cells'][batch][:])
            yield {
                'grids': grids,
                'cells': cells,
                'actions': actions,
                'rewards': rewards,
                'next_grids': next_grids,
                'next_cells': next_cells
            }

    train_gen = BackgroundGenerator(data_gen(0, split), k=10)
    test_gen = BackgroundGenerator(data_gen(split, end), k=10)
    return {
        "n_train_steps": split,
        "n_test_steps": end - split,
        "train_gen": train_gen,
        "test_gen": test_gen
    }


def get_data(self, fname="data-experience-shuffle-sub.npy"):
    data = np.load(fname)
    grids, oh_cells, actions, rewards, next_grids, next_oh_cells = \
        self.prep_data(*map(np.array, zip(*data)))

    split_perc = 0.9  # Percentage of data to train on
    split = int(len(grids) * split_perc) // self.batch_size
    end = len(grids) // self.batch_size

    def data_gen(start, stop):
        for i in range(start, stop):
            batch = slice(i * self.batch_size, (i + 1) * self.batch_size)
            yield {
                'grids': grids[batch],
                'cells': oh_cells[batch],
                'actions': actions[batch],
                'rewards': rewards[batch],
                'next_grids': next_grids[batch],
                'next_cells': next_oh_cells[batch]
            }

    train_gen = data_gen(0, split)
    test_gen = data_gen(split, end)
    return {
        "n_train_steps": split,
        "n_test_steps": end - split,
        "train_gen": train_gen,
        "test_gen": test_gen
    }
