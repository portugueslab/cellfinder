from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import napari
import numpy as np
from napari.utils.io import magic_imread
from pathlib import Path

from PySide2.QtWidgets import QApplication

from imlib.general.system import get_sorted_file_paths, ensure_directory_exists
from imlib.general.list import unique_elements_lists
from imlib.image.metadata import define_pixel_sizes
from imlib.IO.cells import cells_xml_to_df, save_cells, get_cells
from imlib.cells.cells import Cell
from imlib.IO.yaml import save_yaml

from cellfinder.extract.extract_cubes import main as extract_cubes_main
import cellfinder.tools.parser as cellfinder_parse
# from cellfinder.viewer.two_dimensional import estimate_image_max

OUTPUT_NAME = "curated_cells.xml"
CURATED_POINTS = []


def parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = curation_parser(parser)
    parser = cellfinder_parse.pixel_parser(parser)
    parser = cellfinder_parse.misc_parse(parser)
    parser = cellfinder_parse.cube_extract_parse(parser)
    return parser


def curation_parser(parser):
    parser.add_argument(
        dest="signal_image_paths", type=str, help="Signal images"
    )
    parser.add_argument(
        dest="background_image_paths", type=str, help="Background images",
    )

    parser.add_argument(
        dest="cells_xml", type=str, help="Path to the .xml cell file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for curation results",
    )
    parser.add_argument(
        "--symbol", type=str, default="ring", help="Marker symbol."
    )
    parser.add_argument(
        "--marker-size", type=int, default=15, help="Marker size."
    )
    parser.add_argument(
        "--opacity", type=float, default=0.6, help="Opacity of the markers."
    )
    return parser


def get_cell_labels_arrays(
    cells_file, new_order=[2, 1, 0], type_column="type"
):
    df = cells_xml_to_df(cells_file)

    labels = df[type_column]
    labels = labels.to_numpy()
    cells_df = df.drop(columns=[type_column])
    cells = cells_df[cells_df.columns[new_order]]
    cells = cells.to_numpy()

    # convert to boolean
    labels = labels == 2
    return cells, labels


def main():
    args = parser().parse_args()
    args = define_pixel_sizes(args)

    if args.output is None:
        output = Path(args.cells_xml)
        output_directory = output.parent
        print(
            f"No output directory given, so setting output "
            f"directory to: {output_directory}"
        )
    else:
        output_directory = Path(args.output)

    ensure_directory_exists(str(output_directory))
    output_filename = output_directory / "curated_cells.xml"
    start_cube_extraction(args, output_directory, output_filename)


def start_cube_extraction(args, output_directory, output_filename):
    """Extract cubes for training"""

    if not output_filename.exists():
        print(
            "No curation results have been saved. "
            "Please save before extracting cubes"
        )
    else:
        print(f"Saving cubes to: {output_directory}")
        run_extraction(
            output_filename,
            output_directory,
            args.signal_image_paths,
            args.background_image_paths,
            args.cube_depth,
            args.cube_width,
            args.cube_height,
            args.x_pixel_um,
            args.y_pixel_um,
            args.z_pixel_um,
            args.x_pixel_um_network,
            args.y_pixel_um_network,
            args.z_pixel_um_network,
            args.max_ram,
            args.n_free_cpus,
            args.save_empty_cubes,
        )

        print("Saving yaml file to use for training")
        save_yaml_file(output_directory)

        print("Closing window")
        QApplication.closeAllWindows()
        print(
            "Finished! You may now annotate more "
            "datasets, or go straight to training"
        )


def run_extraction(
    output_filename,
    output_directory,
    signal_paths,
    background_paths,
    cube_depth,
    cube_width,
    cube_height,
    x_pixel_um,
    y_pixel_um,
    z_pixel_um,
    x_pixel_um_network,
    y_pixel_um_network,
    z_pixel_um_network,
    max_ram,
    n_free_cpus,
    save_empty_cubes,
):
    planes_paths = {}
    planes_paths[0] = get_sorted_file_paths(
        signal_paths, file_extension=".tif"
    )
    planes_paths[1] = get_sorted_file_paths(
        background_paths, file_extension=".tif"
    )

    all_candidates = get_cells(str(output_filename))

    cells = [c for c in all_candidates if c.is_cell()]
    non_cells = [c for c in all_candidates if not c.is_cell()]

    to_extract = {"cells": cells, "non_cells": non_cells}

    for cell_type, cell_list in to_extract.items():
        print(f"Extracting type: {cell_type}")
        cell_type_output_directory = output_directory / cell_type
        print(f"Saving to: {cell_type_output_directory}")
        ensure_directory_exists(str(cell_type_output_directory))
        extract_cubes_main(
            cell_list,
            cell_type_output_directory,
            planes_paths,
            cube_depth,
            cube_width,
            cube_height,
            x_pixel_um,
            y_pixel_um,
            z_pixel_um,
            x_pixel_um_network,
            y_pixel_um_network,
            z_pixel_um_network,
            max_ram,
            n_free_cpus,
            save_empty_cubes,
        )


def save_yaml_file(output_directory):
    yaml_filename = output_directory / "training.yml"
    yaml_section = [
        {
            "cube_dir": str(output_directory / "cells"),
            "cell_def": "",
            "type": "cell",
            "signal_channel": 0,
            "bg_channel": 1,
        },
        {
            "cube_dir": str(output_directory / "non_cells"),
            "cell_def": "",
            "type": "no_cell",
            "signal_channel": 0,
            "bg_channel": 1,
        },
    ]

    yaml_contents = {"data": yaml_section}
    save_yaml(yaml_contents, yaml_filename)


if __name__ == "__main__":
    main()
