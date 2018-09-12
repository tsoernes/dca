#!/usr/bin/python3
from math import cos, radians, sin, sqrt
from tkinter import Canvas, Frame, Tk

from gridfuncs import GF

label_colors = [
    '#FF2D00', '#FFD800', '#00FF68', '#00FFE4', '#0059FF', '#A200FF', '#FF00F7'
]


class Hexagon:
    def __init__(self, parent, x, y, length, color, outline="gray", top="pointy",
                 tags=""):
        '''
        :param Tk parent
        :param int x: Top left x coordinate
        :param int y: Top left y coordinate
        :param int length: Length of a side
        :param str color: Fill color
        :param str top: 'pointy' or 'flat'
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

        if top == "pointy":
            self.start_angle = 30
        elif top == "flat":
            self.start_angle = 0

        self.shape = None
        self.draw()

    def draw(self):
        start_x = self.x
        start_y = self.y
        angle = 60
        coords = []
        for i in range(6):
            angl = radians(self.start_angle + angle * i)
            end_x = start_x + self.length * cos(angl)
            end_y = start_y + self.length * sin(angl)
            coords.append([start_x, start_y])
            start_x = end_x
            start_y = end_y
        self.shape = self.parent.create_polygon(
            *coords, fill=self.color, outline=self.outline, tags=self.tags)


class HexagonGrid(Frame):
    def __init__(self,
                 parent,
                 rows,
                 cols,
                 size=80,
                 color="#a1e2a1",
                 marked_color="#000000",
                 bg="#ffffff",
                 show_coords=True,
                 show_labels=False,
                 color_labels=True,
                 shape="rhomb",
                 *args,
                 **kwargs):
        '''
                 bg="#a1e2a1",
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
        self.color = color
        self.marked_color = marked_color
        self.font_size = 28

        self.hexagons = []
        if shape == "rect":
            self.left_offset = size / 2
            self.top_offset = 1
            width = size * cols + (cols + 1) * size / 2
            height = (rows + 1 / 2) * sqrt(3) * size + self.top_offset
            self.can = Canvas(self, width=width, height=height, bg=bg)
            self.init_grid_rect(rows, cols, size, show_coords)
        elif shape == "rhomb":
            self.left_offset = size
            self.top_offset = 1
            width = sqrt(3) * size * (rows + cols)
            # width = sqrt(3) * size * (rows + cols / 2)
            height = (rows + 1 / 2) * 1.5 * size + self.top_offset
            self.can = Canvas(self, width=width, height=height, bg=bg)
            self.init_grid_rhomb(rows, cols, size, show_coords, show_labels, color_labels)
        else:
            print(f"Invalid shape {shape}")
            raise Exception

        self.can.pack()
        self.can.bind("<Button-1>", self.onclick)

    def init_grid_rect(self, rows, cols, size, show_coords):
        for r in range(rows):
            hxs = []
            for c in range(cols):
                if c % 2 == 0:
                    offset = size * sqrt(3) / 2
                else:
                    offset = 0
                label = GF.labels[r][c]
                h = Hexagon(
                    self.can,
                    c * (size * 1.5) + self.left_offset,
                    (r * (size * sqrt(3))) + offset + self.top_offset,
                    size,
                    color=label_colors[label],
                    top="flat",
                    tags="{},{}-{}".format(r, c, label))
                hxs.append(h)

                if show_coords:
                    coords = "{}, {}".format(r, c)
                    self.can.create_text(
                        c * (size * 1.5) + (size),
                        (r * (size * sqrt(3))) + offset + size - size / 4,
                        font=("Times", self.font_size, "bold"),
                        text=coords)
            self.hexagons.append(hxs)

    def init_grid_rhomb(self,
                        rows,
                        cols,
                        size,
                        show_coords,
                        show_labels=False,
                        color_labels=True):
        col_offset = 0
        for r in range(rows):
            hxs = []
            for c in range(cols):
                label = GF.labels[r][c]
                lc = label_colors[label] if color_labels else "#C0C0C0"
                h = Hexagon(
                    self.can,
                    c * (size * sqrt(3)) + col_offset + self.left_offset,
                    (r * (size * 1.5)) + self.top_offset,
                    size,
                    color=lc,
                    top="pointy",
                    tags="{},{}-{}".format(r, c, label))
                hxs.append(h)

                if show_coords:
                    coords = "{}, {}".format(r, c)
                    self.can.create_text(
                        c * (size * sqrt(3)) + size + col_offset,
                        (r * (size * 1.5)) + size,
                        text=coords,
                        font=("Times", self.font_size, "bold"))
                if show_labels:
                    self.can.create_text(
                        c * (size * sqrt(3)) + size + col_offset,
                        (r * (size * 1.5)) + size,
                        text=label,
                        font=("Times", self.font_size))
                if (r, c) == (3, 4):
                    self.can.create_text(
                        c * (size * sqrt(3)) + size + col_offset,
                        (r * (size * 1.5)) + size,
                        text="D",
                        font=("Times", self.font_size))
                if (r, c) == (3, 3):
                    self.can.create_text(
                        c * (size * sqrt(3)) + size + col_offset,
                        (r * (size * 1.5)) + size,
                        text="A",
                        font=("Times", self.font_size))
            col_offset += size * sqrt(3) / 2
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
        # self.can.itemconfigure(clicked, fill=self.marked_color)
        tags = self.can.gettags(clicked)[0]
        # TODO find row, col of clicked hexagon
        self.mark_neighs(int(tags[0]), int(tags[2]))
        # print(tags)

    def mark_cell(self, row, col):
        self.can.itemconfigure(self.hexagons[row][col].tags, fill=self.marked_color)

    def unmark_cell(self, row, col):
        h = self.hexagons[row][col]
        self.can.itemconfigure(h.tags, fill=h.color)

    def mark_neighs(self, row, col, c1="#808080", c2="#DCDCDC", c3="#C0C0C0"):
        # if delete_other:
        #     neighs = GF.neighbors2(row, col)
        #     for r, li in enumerate(self.hexagons):
        #         for c, h in enumerate(li):
        #             if (r, c) not in neighs:
        #                 self.can.delete(h.shape)
        h = self.hexagons[row][col]
        self.can.itemconfigure(h.tags, fill=c1)
        for neigh in GF.neighbors(2, row, col):
            h = self.hexagons[neigh[0]][neigh[1]]
            self.can.itemconfigure(h.tags, fill=c2)
        for neigh in GF.neighbors(1, row, col):
            h = self.hexagons[neigh[0]][neigh[1]]
            self.can.itemconfigure(h.tags, fill=c3)

    def hoff_illu(self):
        c1 = '#FFD800'
        self.mark_neighs(3, 3, c1, c1, c1)
        c2 = '#00FF68'
        self.mark_neighs(3, 4, c2, c2, c2)
        # d1 = '#00FFFF'
        # d2 = '#00FE00'
        # h = self.hexagons[3][3]
        # self.can.itemconfigure(h.tags, fill=d1)


class Gui:
    def __init__(self, dims, shape="rhomb"):
        self.root = Tk()
        self.hgrid = HexagonGrid(
            self.root,
            *dims,
            show_coords=False,
            show_labels=False,
            color_labels=False,
            shape=shape)
        self.hgrid.pack()
        self.hgrid.hoff_illu()

    def step(self):
        self.root.update_idletasks()
        self.root.update()

    def test(self):
        self.root.mainloop()


# TODO: Use a gradient color scheme for cells; i.e.
# the more busy a cell is, the darker/denser its color
