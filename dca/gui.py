#!/usr/bin/python3
from tkinter import (Frame, Tk, Canvas)
from math import cos, sin, sqrt, radians

from grid import Grid


label_colors = [
    '#FF2D00',
    '#FFD800',
    '#00FF68',
    '#00FFE4',
    '#0059FF',
    '#A200FF',
    '#FF00F7'
]


class Hexagon:
    def __init__(self, parent, x, y, length, color, outline="gray", tags=""):
        '''
        :param Tk parent
        :param int x: Top left x coordinate
        :param int y: Top left y coordinate
        :param int length: Length of a side
        :param str color: Fill color
        :param str tags: Tags
        '''
        self.parent = parent
        self.x = x
        self.y = y
        self.length = length
        self.color = color
        self.outline = outline
        self.tags = tags
        self.selected = False

        self.shape = None
        self.draw()

    def draw(self):
        start_x = self.x
        start_y = self.y
        angle = 60
        coords = []
        for i in range(6):
                end_x = start_x + self.length * cos(radians(angle * i))
                end_y = start_y + self.length * sin(radians(angle * i))
                coords.append([start_x, start_y])
                start_x = end_x
                start_y = end_y
        self.shape = self.parent.create_polygon(
                *coords,
                fill=self.color,
                outline=self.outline,
                tags=self.tags)


class HexagonGrid(Frame):
    def __init__(self, parent, rows, cols, labels, size=50,
                 color="#a1e2a1", marked_color="#000000", bg="#a1e2a1",
                 show_coords=False,
                 *args, **kwargs):
        '''
        :param Tk parent
        :param int cols
        :param int rows
        :param np.array(rows,cols) labels: cell labels. Useful for visualizing
            channel reuse distance (even for strats that don't partition chs)
        :param int size: Hexagon line length
        :param str color: Hexagon fill color
        :param str marked_color: Hexagon fill color when selected
        :param bg: Background color
        '''
        Frame.__init__(self, parent)
        self.left_offset = size/2
        self.top_offset = 1
        width = size*cols + (cols+1)*size/2
        height = (rows+1/2)*sqrt(3)*size + self.top_offset
        self.can = Canvas(self, width=width, height=height, bg=bg)
        self.can.pack()

        self.labels = labels
        self.color = color
        self.marked_color = marked_color

        self.hexagons = []
        self.initGrid(rows, cols, size, show_coords)

        self.can.bind("<Button-1>", self.onclick)

    def initGrid(self, rows, cols, size, show_coords):
        for r in range(rows):
            hxs = []
            for c in range(cols):
                if c % 2 == 0:
                    offset = size * sqrt(3) / 2
                else:
                    offset = 0
                label = self.labels[r][c]
                h = Hexagon(self.can,
                            c * (size * 1.5) + self.left_offset,
                            (r * (size * sqrt(3))) + offset + self.top_offset,
                            size,
                            color=label_colors[label],
                            tags="{},{}-{}".format(r, c, label))
                hxs.append(h)

                if show_coords:
                    coords = "{}, {}".format(r, c)
                    self.can.create_text(
                            c * (size * 1.5) + (size),
                            (r * (size * sqrt(3))) + offset + (size/3),
                            text=coords)
            self.hexagons.append(hxs)

    def onclick(self, evt):
        """
        hexagon detection on mouse click
        """
        x, y = evt.x, evt.y
        clicked = self.can.find_closest(x, y)  # find closest object
        if self.can.type(clicked) != "polygon":
            return
        # Unselect hexagons and revert to their default color
        for li in self.hexagons:
            for h in li:
                self.can.itemconfigure(h.tags, fill=h.color)
        self.can.itemconfigure(clicked, fill=self.marked_color)
        # print(self.can.gettags(clicked)[0])

    def mark_cell(self, row, col):
        self.can.itemconfigure(
                self.hexagons[row][col].tags, fill=self.marked_color)

    def unmark_cell(self, row, col):
        h = self.hexagons[row][col]
        self.can.itemconfigure(h.tags, fill=h.color)


class Gui:
    def __init__(self, grid, keydown):
        self.grid = grid
        self.root = Tk()
        self.hgrid = HexagonGrid(
                self.root, grid.rows, grid.cols,
                grid.labels, show_coords=True)
        # Call 'keydown' func for graceful exit on 'q' key
        # or on window close
        self.root.bind("q", keydown)
        self.root.protocol("WM_DELETE_WINDOW", keydown)
        self.hgrid.pack()

    def step(self):
        self.root.update_idletasks()
        self.root.update()

    def test(self):
        self.root.mainloop()

    def neighs(self, radius=1, grayscale=False):
        x, y = 4, 4
        neighs = [(x, y)]
        if radius == 1:
            neighs += self.grid.neighbors1(x, y)
        else:
            neighs += self.grid.neighbors2(x, y)
        for r, li in enumerate(self.hgrid.hexagons):
            for c, h in enumerate(li):
                if (r, c) not in neighs:
                    self.hgrid.can.delete(h.shape)
        if grayscale:
            # TODO Find colors
            for neigh in self.grid.neighbors1(x, y):
                h = self.hgrid.hexagons[neigh[0]][neigh[1]]
                self.hgrid.can.itemconfigure(h.tags, fill="")
            for neigh in self.grid.neighbors2(x, y):
                h = self.hgrid.hexagons[neigh[0]][neigh[1]]
                self.hgrid.can.itemconfigure(h.tags, fill="")


# TODO: Use a gradient color scheme for cells; i.e.
# the more busy a cell is, the darker/denser its color
if __name__ == '__main__':
    g = Grid(7, 7, 70)
    gui = Gui(g, None)
    gui.neighs(1)
    gui.test()
