import os
import threading
from queue import Queue

import numpy as np
import h5py


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, k=1):
        """
        Take a generator and return a iterator that prefetches the generator
        in a separate thread.

        k: Number of objects to prefetch and hold in memory
        """
        threading.Thread.__init__(self)
        # Tell Python it's OK to exit even if this thread has not finished
        self.daemon = True
        self.queue = Queue.Queue(k)
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
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


def h5py_save(fname, states, cells, chs, rewards, new_states, new_cells,
              chunk_size=100000):
    """
    chunk_size: Number of experience tuples to load at a time
    """
    n_fname = next_filename(fname)
    with h5py.File(n_fname, "x") as f:
        def create_ds(name, shape, dtype):
            return f.create_dataset(
                name, (len(states), *shape), maxshape=(None, *shape),
                dtype=dtype, chunks=(chunk_size, *shape))
        ds_states = create_ds("states", (7, 7, 70), np.int8)
        ds_cells = create_ds("cells", (2,), np.int8)
        ds_chs = create_ds("chs", (), np.int8)
        ds_rewards = create_ds("rewards", (), np.int32)
        ds_new_states = create_ds("new_states", (7, 7, 70), np.bool)
        ds_new_cells = create_ds("new_cells", (2,), np.int8)
        ds_states[:] = states
        ds_cells[:] = cells
        ds_chs[:] = chs
        ds_rewards[:] = rewards
        ds_new_states[:] = new_states
        ds_new_cells[:] = new_cells
    print(f"Wrote {len(states)} experience tuples to {n_fname}")


def h5py_concat(paths):
    path = None
    with h5py.File(path, "a") as f:
        dset = f.create_dataset('voltage284', (10**5,), maxshape=(None,),
                                dtype='i8', chunks=(10**4,))
        dset[:] = np.random.random(dset.shape)
        print(dset.shape)
        # (100000,)

        for i in range(3):
            dset.resize(dset.shape[0] + 10**4, axis=0)
            dset[-10**4:] = np.random.random(10**4)
            print(dset.shape)
