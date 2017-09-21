
class FCState():
    def __init__(self):
        pass

    # Fully connected network; attempted replica of Singh
    def _state_frepr(self, state):
        """
        Feature representation of a state
        """
        frepr = np.zeros((self.rows, self.cols, self.n_channels+1))
        # Number of available channels for each cell
        frepr[:, :, -1] = self.n_channels - np.sum(self.state, axis=2)
        # The number of times each channel is used within a 1 cell radius,
        # not including self
        for i in range(self.rows):
            for j in range(self.cols):
                for ch in range(self.n_channels):
                    neighs = self.neighbors2(i, j)
                    for neigh in neighs:
                        frepr[i][j][ch] += self.state[neigh[0]][neigh[1]][ch]
        return frepr


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
