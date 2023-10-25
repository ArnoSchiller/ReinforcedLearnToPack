import os
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.patches import Rectangle
from PIL import Image

from gym_packing.data.bin import Bin
from gym_packing.data.packing_variant import PackingVariant


class PackingVisualization():
    def __init__(self):
        self.colors = list(mcolors.TABLEAU_COLORS.keys())

    def get_color(self, index: int) -> str:
        color = self.colors[index % len(self.colors)]
        return color

    def visualize_bin(
        self, bin: Bin,
        snappoint_min_z: 'int|None' = None,
        title: str = None,
        show: bool = True,
        output_dir: 'str | None' = None,
        return_png: bool = False
    ):
        is_2d, _ = bin.is_packing_2d()
        if is_2d:
            plot_fn = self._visualize_bin_2d
        else:
            plot_fn = self._visualize_bin_3d

        return plot_fn(
            bin=bin,
            snappoint_min_z=snappoint_min_z,
            title=title,
            show=show,
            output_dir=output_dir,
            return_png=return_png
        )

    def visualize_packing_variant(
        self,
        variant: PackingVariant,
        show: bool = True,
        output_dir: 'str | None' = None,
        return_png=False
    ):
        images = []
        for idx, bin in enumerate(variant.bins):
            img = self.visualize_bin(
                bin=bin,
                title=f"Bin {idx+1}",
                show=show,
                output_dir=output_dir,
                return_png=return_png
            )
            images.append(img)
        return images

    def _visualize_bin_2d(
        self,
        bin: Bin,
        snappoint_min_z: 'int|None' = None,
        title: str = None,
        show: bool = True,
        output_dir: 'str | None' = None,
        return_png: bool = False
    ):
        is_2d, dimensions = bin.is_packing_2d()
        if not is_2d:
            raise ValueError("Bin is not 2D, use _visualize_bin_3d.")

        items = bin.packed_items
        fig, ax = plt.subplots()

        pos_dim_2d = [
            item.to_position_and_dimension_2d(dimensions) for item in items
        ]
        x_max, y_max = bin.get_dimension_2d(dimensions)

        ax.axes.set_xlim(0, x_max)
        ax.axes.set_ylim(0, y_max)
        ax.set_xlabel(dimensions[0])
        ax.set_ylabel(dimensions[1])

        if title is not None:
            ax.set_title(title)
        ax.set_aspect('equal')

        for idx, (pos, dim) in enumerate(pos_dim_2d):
            ax.add_patch(
                Rectangle(pos, dim[0], dim[1],
                          facecolor=self.get_color(idx), edgecolor="black", linewidth=2)
            )

        if snappoint_min_z is not None:
            snappoints = bin.get_snappoints(snappoint_min_z)

            radius = 0.2
            for point in snappoints:
                ax.add_patch(
                    plt.Circle((point.x, point.z), radius, color='r')
                )

        if not output_dir is None:
            os.makedirs(output_dir, exist_ok=True)
            name = "packing_%s.png"
            number = 0
            while os.path.exists(os.path.join(output_dir, name % number)):
                number += 1
            plt.savefig(os.path.join(output_dir, name % number))

        if show:
            plt.show()

        if return_png:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img = Image.open(img_buf)
            img_buf.close()
            return img

        return fig

    def _visualize_bin_3d(
        self,
        bin: Bin,
        snappoint_min_z: 'int|None' = None,
        title: str = None,
        show: bool = True,
        output_dir: 'str | None' = None,
        return_png: bool = False
    ):
        items = bin.packed_items
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.axes.set_xlim3d(0, bin.width)
        ax.axes.set_ylim3d(0, bin.length)
        ax.axes.set_zlim3d(0, bin.height)

        if title is not None:
            ax.set_title(title)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

        """
        # draw the pallet
        step = 2
        pallet_height = 0.01
        for width_step in range(0, self.container_size[0], step):
            ax.bar3d(
                width_step,
                0,
                0,
                2,
                self.container_size[1],
                pallet_height,
                color="#e7cfb4",
                edgecolor="black",
            )
        """
        for item in items:
            ax.bar3d(
                item.position.x,
                item.position.y,
                item.position.z,
                item.width,
                item.length,
                item.height,
                alpha=0.8,
                edgecolor="black",
            )

        ax.set_xlabel("Width")
        ax.set_ylabel("Length")
        ax.set_zlabel("Height")

        if not output_dir is None:
            os.makedirs(output_dir, exist_ok=True)
            name = "packing_%s.png"
            number = 0
            while os.path.exists(os.path.join(output_dir, name % number)):
                number += 1
            plt.savefig(os.path.join(output_dir, name % number))

        if show:
            plt.show()

        if return_png:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img = Image.open(img_buf)
            img_buf.close()
            return img

        return fig
