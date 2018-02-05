import os
import threading
from queue import Queue

import h5py
import numpy as np

from nets.utils import prep_data


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, n_prefetch=1):
        """
        Take a generator and return a iterator that prefetches the generator
        in a separate thread.

        k: Number of objects to prefetch and hold in memory
        """
        threading.Thread.__init__(self)
        # Tell Python it's OK to exit even if this thread has not finished
        self.daemon = True
        self.queue = Queue(n_prefetch)
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item


def shuffle_in_unison(arrs):
    """
    Shuffle the arrays in 'arrs' inplace and in unison.
    """
    rng_state = np.random.get_state()
    for arr in arrs:
        np.random.set_state(rng_state)
        np.random.shuffle(arr)


def h5py_shuffle_in_unison(fname="data-experience.0"):
    """
    Shuffle a hdf5 dataset inplace and in unison. fname without file ext.
    inplace means that it doesnt load everything into ram.
    """
    with h5py.File(fname + ".hdf5", "r+", driver="core") as f:
        rng_state = np.random.get_state()
        for key in f.keys():
            np.random.set_state(rng_state)
            np.random.shuffle(f[key])


def next_filename(fname, ext=".hdf5"):
    """
    For a given filename 'fname' (without extension), and extension 'ext',
    return 'fname.n.ext' where n is the lowest integer that does not
    already exists on the file system
    """

    def next_fname(num):
        return fname + "." + str(num) + ext

    data_set_number = 0
    n_fname = next_fname(data_set_number)
    while os.path.isfile(n_fname):
        data_set_number += 1
        n_fname = next_fname(data_set_number)
    return n_fname


def h5py_save(fname,
              grids,
              cells,
              chs,
              rewards,
              next_grids=None,
              next_cells=None,
              chunk_size=100000):
    """
    fname: file name without extension
    chunk_size: Number of experience tuples to load at a time
    """
    n_fname = next_filename(fname)
    with h5py.File(n_fname, "x") as f:

        def create_ds(name, shape, dtype):
            return f.create_dataset(
                name, (len(grids), *shape),
                maxshape=(None, *shape),
                dtype=dtype,
                chunks=(chunk_size, *shape))

        n_rows, n_cols, n_channels = grids[0].shape
        ds_grids = create_ds("grids", (n_rows, n_cols, n_channels), np.bool)
        ds_cells = create_ds("cells", (2, ), np.int8)
        ds_chs = create_ds("chs", (), np.int8)
        ds_rewards = create_ds("rewards", (), np.int32)
        if next_grids is not None:
            ds_next_grids = create_ds("next_grids", (n_rows, n_cols, n_channels), np.bool)
        if next_cells is not None:
            ds_next_cells = create_ds("next_cells", (2, ), np.int8)
        ds_grids[:] = grids
        ds_cells[:] = cells
        ds_chs[:] = chs
        ds_rewards[:] = rewards
        if next_grids is not None:
            ds_next_grids[:] = next_grids
        if next_cells is not None:
            ds_next_cells[:] = next_cells
    print(f"Wrote {len(grids)} experience tuples to {n_fname}")


def h5py_save_append(fname,
                     grids,
                     cells,
                     chs,
                     rewards,
                     next_grids=None,
                     next_cells=None,
                     chunk_size=100000):
    """
    Append to existing data set. See 'h5py_save'.
    """
    efname = fname + ".0.hdf5"
    if not os.path.isfile(efname):
        print(f"File not found {efname}, creating new")
        h5py_save(fname, grids, cells, chs, rewards, next_grids, next_cells)
        return
    n = len(grids)
    with h5py.File(efname, "r+") as f:
        for ds_key in f.keys():
            dset = f[ds_key]
            dset.resize(dset.shape[0] + n, axis=0)
        f['grids'][-n:] = grids
        f['cells'][-n:] = cells
        f['chs'][-n:] = chs
        f['rewards'][-n:] = rewards
        if next_grids is not None:
            f['next_grids'][-n:] = next_grids
        if next_cells is not None:
            f['next_cells'][-n:] = next_cells
    print(f"Appended {len(grids)} experience tuples to {efname}")


def get_data_h5py(batch_size, fname="data-experience.0", split_perc=0.9, n_prefetch=50):
    """
    fname: file name without extension
    split_perc: Percentage of data to train on
    n_prefetch: number of batches to load in RAM

    Return (n_tr, n_te, g_tr, g_te) where
    n_tr: number of training samples
    n_te: number of test samples
    g_tr: generator that yields training data
    g_te: generator that yields test data
    """
    # Open file handle, but don't load contents into memory
    h5f = h5py.File(fname + ".hdf5", "r")
    entries = len(h5f['grids'])
    split = int(entries * split_perc) // batch_size
    end = entries // batch_size
    has_next_state = 'next_grids' in h5f

    def data_gen(start, stop):
        for i in range(start, stop):
            batch = slice(i * batch_size, (i + 1) * batch_size)
            # Load batch data into memory and prep it
            cells = list(map(tuple, h5f['cells'][batch][:]))
            res = {}
            if has_next_state:
                next_grids = h5f['next_grids'][batch][:]
                next_cells = list(map(tuple, h5f['next_cells'][batch][:]))
            else:
                next_grids = None
                next_cells = None
            prepped = \
                prep_data(
                    h5f['grids'][batch][:],
                    cells,
                    h5f['chs'][batch][:],
                    h5f['rewards'][batch][:],
                    next_grids,
                    next_cells)
            if has_next_state:
                pgrids, oh_cells, pactions, prewards, pnext_grids, oh_next_cells = prepped
            else:
                pgrids, oh_cells, pactions, prewards = prepped
            res = {
                'grids': pgrids,
                'cells': cells,
                'oh_cells': oh_cells,
                'actions': pactions,
                'rewards': prewards,
            }
            if has_next_state:
                res.update({
                    'next_grids': pnext_grids,
                    'next_cells': next_cells,
                    'oh_next_cells': oh_next_cells
                })
            yield res

    train_gen = BackgroundGenerator(data_gen(0, split), n_prefetch)
    test_gen = BackgroundGenerator(data_gen(split, end), n_prefetch)
    return {
        "n_train_steps": split,
        "n_test_steps": end - split,
        "train_gen": train_gen,
        "test_gen": test_gen
    }


def get_data(batch_size, fname="data-experience-shuffle-sub.npy"):
    data = np.load(fname)
    grids, oh_cells, actions, rewards, next_grids, next_oh_cells = \
        prep_data(*map(np.array, zip(*data)))

    split_perc = 0.9  # Percentage of data to train on
    split = int(len(grids) * split_perc) // batch_size
    end = len(grids) // batch_size

    def data_gen(start, stop):
        for i in range(start, stop):
            batch = slice(i * batch_size, (i + 1) * batch_size)
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


if __name__ == "__main__":
    h5py_shuffle_in_unison()
