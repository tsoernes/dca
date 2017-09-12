#!/usr/bin/python3
from tkinter import (Frame, Tk, Canvas)
from math import cos, sin, sqrt, radians
from main import partition_cells


label_colors = [
    '#FF2D00',
    '#FFD800',
    '#00FFC9',
    '#00F0FF',
    '#0059FF',
    '#A200FF',
    '#FF009E'
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
        self.parent.create_polygon(*coords,
                                   fill=self.color,
                                   outline=self.outline,
                                   tags=self.tags)


class HexagonGrid(Frame):
    def __init__(self, parent, rows, cols, labels, size=50,
                 color="#a1e2a1", marked_color="#53ca53", bg="#a1e2a1",
                 show_coords=False,
                 *args, **kwargs):
        '''
        :param Tk parent
        :param int cols
        :param int rows
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
        print(self.can.gettags(clicked)[0])


if __name__ == '__main__':
    root = Tk()
    rows = 10
    cols = 8
    labels = partition_cells(rows, cols)
    hgrid = HexagonGrid(root, rows, cols, labels, show_coords=True)
    hgrid.pack()
    hgrid.can.itemconfigure(hgrid.hexagons[2][3].tags, fill=hgrid.marked_color)
    root.mainloop()
