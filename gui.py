#!/usr/bin/python3
from tkinter import (Frame, Tk, Canvas)
from math import cos, sin, sqrt, radians


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
        self.parent.create_polygon(coords[0][0],
                                   coords[0][1],
                                   coords[1][0],
                                   coords[1][1],
                                   coords[2][0],
                                   coords[2][1],
                                   coords[3][0],
                                   coords[3][1],
                                   coords[4][0],
                                   coords[4][1],
                                   coords[5][0],
                                   coords[5][1],
                                   fill=self.color,
                                   outline=self.outline,
                                   tags=self.tags)


class HexagonGrid(Frame):
    def __init__(self, parent, cols, rows, size=30,
                 color="#a1e2a1", marked_color="#53ca53", bg="#a1e2a1",
                 *args, **kwargs):
        '''
        :param Tk parent
        :param int cols
        :param int rows
        :param int size: Hexagon line length
        :param str color: Hexagon fill color
        :param str marked_color: Hexagon fill color when selected
        '''
        Frame.__init__(self, parent, )
        self.left_offset = size/2
        self.top_offset = 1
        width = size*cols + (cols+1)*size/2
        height = size*(rows+3) + (rows+1)*size/2
        height = (rows+1/2)*sqrt(3)*size + self.top_offset
        self.can = Canvas(self, width=width, height=height, bg=bg)
        self.can.pack()

        self.color = color
        self.marked_color = marked_color

        self.hexagons = []
        self.initGrid(cols, rows, size, debug=False)

        self.can.bind("<Button-1>", self.click)

    def initGrid(self, cols, rows, size, debug):
        for c in range(cols):
            if c % 2 == 0:
                offset = size * sqrt(3) / 2
            else:
                offset = 0
            for r in range(rows):
                h = Hexagon(self.can,
                            c * (size * 1.5) + self.left_offset,
                            (r * (size * sqrt(3))) + offset + self.top_offset,
                            size,
                            color=self.color,
                            tags="{}.{}".format(r, c))
                self.hexagons.append(h)

                if debug:
                    coords = "{}, {}".format(r, c)
                    self.can.create_text(
                            c * (size * 1.5) + (size/2),
                            (r * (size * sqrt(3))) + offset + (size/2),
                            text=coords)

    def click(self, evt):
        """
        hexagon detection on mouse click
        """
        x, y = evt.x, evt.y
        # Unselect unclicked hexagons and revert to their default color
        for i in self.hexagons:
                i.selected = False
                self.can.itemconfigure(i.tags, fill=i.color)
        clicked = self.can.find_closest(x, y)[0]  # find closest
        hexagon = self.hexagons[int(clicked)-1]
        hexagon.selected = True
        self.can.itemconfigure(hexagon.tags, fill=self.marked_color)


if __name__ == '__main__':
    root = Tk()
    hgrid = HexagonGrid(root, 10, 8).pack()
    root.mainloop()
