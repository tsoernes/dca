"""
An incomplete implementation of BDCL: Borrow with Directional Channel Locking
"""

class Direction(Enum):
    NE = 0
    SE = 1
    S = 2
    SW = 3
    NW = 4
    N = 5


class BDCLGrid(FixedGrid):
    @staticmethod
    def move(row, col, direction):
        raise NotImplementedError
        f = None
        if direction == Direction.NE:
            f = Grid.move_n
        elif direction == Direction.SE:
            f = Grid.move_se
        elif direction == Direction.S:
            f = Grid.move_s
        elif direction == Direction.SW:
            f = Grid.move_sw
        elif direction == Direction.NW:
            f = Grid.move_nw
        elif direction == Direction.N:
            f = Grid.move_n
        return f(row, col)

    @staticmethod
    def direction(from_r, from_c, to_r, to_c):
        raise NotImplementedError
        if from_r - 1 == to_r and from_c == to_c:
            return Direction.N
        if from_r + 1 == to_r and from_c == to_c:
            return Direction.S
        if from_c % 2 == 0:
            if from_r == to_r and from_c + 1 == to_c:
                return Direction.NE
            if from_r + 1 == to_r and from_c + 1 == to_c:
                return Direction.SE
            if from_r + 1 == to_r and from_c - 1 == to_c:
                return Direction.SW
            if from_r == to_r and from_c - 1 == to_c:
                return Direction.NW
        else:
            if from_r - 1 == to_r and from_c + 1 == to_c:
                return Direction.NE
            if from_r == to_r and from_c + 1 == to_c:
                return Direction.SE
            if from_r == to_r and from_c - 1 == to_c:
                return Direction.SW
            if from_r - 1 == to_r and from_c - 1 == to_c:
                return Direction.NW

    @staticmethod
    def move_n(row, col):
        return (row - 1, col)

    @staticmethod
    def move_ne(row, col):
        if col % 2 == 0:
            return (row, col + 1)
        else:
            return (row - 1, col + 1)

    @staticmethod
    def move_se(row, col):
        if col % 2 == 0:
            return (row + 1, col + 1)
        else:
            return (row, col + 1)

    @staticmethod
    def move_s(row, col):
        return (row + 1, col)

    @staticmethod
    def move_sw(row, col):
        if col % 2 == 0:
            return (row + 1, col - 1)
        else:
            return (row, col - 1)

    @staticmethod
    def move_nw(row, col):
        if col % 2 == 0:
            return (row, col - 1)
        else:
            return (row - 1, col - 1)

    @staticmethod
    def distance(r1, c1, r2, c2):
        raise NotImplementedError
        if c1 % 2 == 0:
            if c2 % 2 == 0:
                distance = math.abs()
            else:
                pass
        else:
            if c2 % 2 == 0:
                pass
            else:
                pass
        return distance

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()
        super().__init__(*args, **kwargs)
        # For each channel, a direction is locked if entry is True
        self.locks = np.zeros((self.rows, self.cols, self.n_channels, 7), dtype=bool)
        # A cell and a channel is locked by cell coordinates in entry
        # (-1, -1) if not locked.
        self.locked_by = np.zeros((self.rows, self.cols, self.n_channels, 2), dtype=int)

    def cochannel_cells(self, cell, cell_neigh):
        """
        A cochannel cell of a cell 'c' given a neighbor 'b'
        is a cell with the same label as 'c' within a radius
        of 2 from 'b', i.e. the cells within the channel reuse distance
        of 3 when 'b' borrows a channel from 'c'.
        """
        raise NotImplementedError  # perhaps not implemented correctly
        neighs = self.neighbors2(*cell_neigh)
        coch_cells = []
        label = self.labels[cell]
        for neigh in neighs:
            if self.labels[cell_neigh] == label and neigh != cell:
                coch_cells.append(neigh)
        return coch_cells

    def borrow(self, from_cell, to_cell, ch):
        """
        Borrow a channel 'ch' from cell 'from' to cell 'to'.
        """
        # TODO implementation assumes borrowing allowed, i.e. does not
        # break any constraints or invariants
        coch_cells = self.cochannel_cells(*from_cell, *to_cell)
        for ccell in coch_cells:
            self.locked_by[ccell][ch] = [*to_cell]
        # Cochannel cells with their respective 1-radius neighbors
        coch_cells_wneighs = zip(coch_cells, map(self.neighbors1, coch_cells))
        to_cell_neighs = self.neighbors2(*to_cell)
        # Cells that would violate channel reuse constraint unless
        # locked from borrowing
        for (cc, ccn) in coch_cells_wneighs:
            for tcn in to_cell_neighs:
                if ccn == tcn:
                    for direction in Grid.direction(*cc, *ccn):
                        self.locks[cc][ch][direction] = 1


class BDCLStrat(Strat):
    # Borrowing with Directional Channel Locking (BDCL) of Zhang & Yum (1989).
    def __init__(self, *args, **kwargs):
        super(BDCLStrat, self).__init__(*args, **kwargs)
        self.grid.assign_chs()

    def fn_new(self, row, col):
        """
        ON NEW CALL:
        """
        ch = -1
        # If a nominal channel is available when a call arrives in a cell,
        # the smallest numbered such channel is assigned to the call.
        for idx, isNom in enumerate(self.grid.nom_chs[row][col]):
            if isNom and self.state[row][col][idx]:
                ch = idx
                break
        if ch != -1:
            return ch

        # If no nominal channel is available, then the largest numbered
        # free channel is borrowed from the neighbour with
        # the most free channels.

        # If all the nominal channels are busy, search
        # through all the neighboring cells of P to identify all the free
        # channels as well as all the “locked” channels but with cell P
        # in the nonlocking direction. Call this set of channels X . If X
        # is empty, block the call.
        neighs = self.neighbors1()
        x = []
        for neigh in neighs:
            for chan, inUse in enumerate(self.grid.state[neigh]):
                direction = Grid.direction(*neigh, row, col)
                if not inUse and not self.grid.locks[neigh][ch][direction]:
                    x.append((neigh, chan))
        # Select the channels in X which are either
        # a) free in their two nearby cochannel cells, or
        # b) being locked but the mini-mum distance between cell P
        # and the locking cells is at least three cell units apart.
        # Call the set of selected channels Y .
        # If Y is empty, block the call.
        y = []
        for neigh, chan in x:
            cooch_cells = self.grid.cochannel_cells(row, col, *neigh)

        # 4) The MTSO assigns the particular channel in Y which is
        # the last ordered channel from the cell with the maximum num-
        # ber of free channels. Denote the assigned channel as channel
        # A.
        # 5) With channel x assigned, the three nearby cochannel
        # cells will lock channel x in the appropriate directions. Move
        # channel x from the FC list to the LC lists of the three cells.
        # Cell P is also recorded in LC to indicate that it is responsible

        # When a channel is borrowed, careful accounting of the directional
        # effect of which cells can no longer use that channel because
        # of interference is done.
        # The call is blocked if there are no free channels at all.

        # Changing state (assigning call to cell and ch) to the
        # incoming call should not be done here, only rearrangement
        # of existing calls
        # for locking channel x .
        return ch

    def fn_end(self):
        """
        ON CALL TERM:
        When a call terminates in a cell and the channel so freed is a nominal
        channel, say numbered i, of that cell, then if there is a call
        in that cell on a borrowed channel, the call on the smallest numbered
        borrowed channel is reassigned to i and the borrowed channel
        is returned to the
        appropriate cell. If there is no call on a borrowed channel,
        then if there is a call on a nominal channel numbered larger than i,
        the call on the highest numbered nominal channel is reassigned to i.
        If the call just terminated was itself on a borrowed channel, the
        call on the smallest numbered borrowed channel is reassigned to it
        and that
        channel is returned to the cell from which it was borrowed.
        Notice that when a
        borrowed channel is returned to its original cell, a nominal
        channel becomes
        free in that cell and triggers a reassignment.
        """
        pass
