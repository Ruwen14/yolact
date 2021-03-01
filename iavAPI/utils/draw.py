import random

def getColorMap(length: int = None, shuffle: bool = False):
    COLORS = [(244, 67, 54),
              (233, 30, 99),
              (156, 39, 176),
              (103, 58, 183),
              (63, 81, 181),
              (33, 150, 243),
              (3, 169, 244),
              (0, 188, 212),
              (0, 150, 136),
              (76, 175, 80),
              (139, 195, 74),
              (205, 220, 57),
              (255, 235, 59),
              (255, 193, 7),
              (255, 152, 0),
              (255, 87, 34),
              (121, 85, 72),
              (158, 158, 158),
              (96, 125, 139)]

    if shuffle:
        random.shuffle(COLORS)

    if length is not None:
        assert isinstance(length, int), "arg length has to be an integer"
        assert 0<= length <= len(COLORS), f'Choose length between {0} and {len(COLORS)}'

        return COLORS[:length]

    return COLORS

def fix_color(number_det, colors):
    """
    Temporary Fix for rotated Bounding Boxex to have same color like text
    """
    number_colors = len(colors)
    COLORS = getColorMap()
    if number_det > number_colors:
        diff = number_det - number_colors
        [colors.append(random.choice(COLORS)) for i in range(diff)]
        return colors

    elif number_det < number_colors:
        diff  =number_colors - number_det
        return colors[:-diff] #Shorten by ne

    else:
        return colors


class PlotUtils(object):

    @staticmethod
    def moving_average(scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    @staticmethod
    def adjust_ligthness(color, amount=0.5):
        """
        Lightens the given color by multiplying (1-luminosity) by the given amount.
        Input can be matplotlib color string, hex string, or RGB tuple.

        Examples:
        >> lighten_color('g', 0.3)
        >> lighten_color('#F034A3', 0.6)
        >> lighten_color((.3,.55,.1), 0.5)
        """
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    @staticmethod
    def get_colours(style: str = 'default', shuffle :bool =False,amount:int = 12):
        """
        Gives colours from a choosen style.
        Args:
            style: style of colour palette;
                Passing 'None' plots all possible colours

            shuffle: bool, Whether to randomly shuffle list before returning

        Return:
            returns list of colours
        """
        from matplotlib import cm
        import random
        cases = {
            'default': ['#2CBDFE', '#47DBCD', '#47DBCD', '#F3A0F2','#9D2EC5', '#661D98', '#F5B14C'],
            'dark_palette': ['#003f5c','#2f4b7c','#665191','#a05195','#d45087','#f95d6a','#ff7c43','#ffa600'],
            'line_plot_5x':  ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a'],
            'plotly_default':['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'],
            'stole_from_matlab':[(0.55, 0.83, 0.78), (1.0, 0.93, 0.44), (0.75, 0.73, 0.85), (0.98, 0.5, 0.45), (0.5, 0.69, 0.83), (0.99, 0.71, 0.38),
                                 (0.7, 0.87, 0.41), (0.74, 0.5, 0.74), (0.85, 0.85, 0.85), (0.8, 0.92, 0.77), (0.99, 0.8, 0.9), (1.0, 1.0, 0.7)],

            'sick_colors' :['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499'],
            'viridis':cm.get_cmap('viridis', amount).colors,
            'inferno': cm.get_cmap('inferno', amount).colors,
            'plasma': cm.get_cmap('plasma', amount).colors,
            'magma': cm.get_cmap('magma', amount).colors,
            'cividis': cm.get_cmap('cividis', amount).colors,
        }


        if style is 'None':
            import matplotlib.pyplot as plt
            print('Showing possible colour lists')
            for case in cases:
                color_list = cases.get(case)
                PlotUtils.plot_colortable(color_list,title=case,shuffle=shuffle)
            print(f'Possible colors are {[case for case in cases]}')
            plt.show()
            return None


        color_list = cases.get(style)
        assert color_list is not None, f'Couldnt find color\n Possible colors are{[case for case in cases]}'


        random.shuffle(color_list) if shuffle else None

        return color_list

    @staticmethod
    def plot_colortable(colors, title=None, shuffle=False, emptycols=0):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import random

        random.shuffle(colors) if shuffle else None



        cell_width = 212
        cell_height = 22
        swatch_width = 48
        margin = 12
        topmargin = 40


        n = len(colors)
        ncols = 4 - emptycols
        nrows = n // ncols + int(n % ncols > 0)

        width = cell_width * 4 + 2 * margin
        height = cell_height * nrows + margin + topmargin
        dpi = 72

        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.subplots_adjust(margin / width, margin / height,
                            (width - margin) / width, (height - topmargin) / height)
        ax.set_xlim(0, cell_width * 4)
        ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_axis_off()
        ax.set_title(title, fontsize=24, loc="left", pad=10)

        for i, name in enumerate(colors):
            row = i % nrows
            col = i // nrows
            y = row * cell_height

            swatch_start_x = cell_width * col
            swatch_end_x = cell_width * col + swatch_width
            text_pos_x = cell_width * col + swatch_width + 7
            try:
                name = [round(n, 2) for n in name]

            except:
                pass
            ax.text(text_pos_x, y, name, fontsize=14,
                    horizontalalignment='left',
                    verticalalignment='center')

            ax.hlines(y, swatch_start_x, swatch_end_x,
                      color=name, linewidth=18)

        return fig





