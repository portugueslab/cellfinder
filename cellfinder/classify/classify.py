import logging
import numpy as np
from imlib.general.system import get_sorted_file_paths, get_num_processes
from imlib.cells.cells import Cell

from imlib.IO.cells import save_cells
from cellfinder.classify.tools import get_model
from cellfinder.classify.cube_generator import CubeGeneratorFromFile
from cellfinder.train.train_yml import models
from cellfinder.detect.filters.post_classification_filters.proximity_filtering import proximity_filter

def main(args, max_workers=3):
    signal_paths = args.signal_planes_paths[args.signal_channel]
    background_paths = args.background_planes_path[0]
    signal_images = get_sorted_file_paths(signal_paths, file_extension="tif")
    background_images = get_sorted_file_paths(
        background_paths, file_extension="tif"
    )

    # Too many workers doesn't increase speed, and uses huge amounts of RAM
    workers = get_num_processes(
        min_free_cpu_cores=args.n_free_cpus, n_max_processes=max_workers
    )

    logging.debug("Initialising cube generator")
    inference_generator = CubeGeneratorFromFile(
        args.paths.cells_file_path,
        signal_images,
        background_images,
        batch_size=args.batch_size,
        x_pixel_um=args.x_pixel_um,
        y_pixel_um=args.y_pixel_um,
        z_pixel_um=args.z_pixel_um,
        x_pixel_um_network=args.x_pixel_um_network,
        y_pixel_um_network=args.y_pixel_um_network,
        z_pixel_um_network=args.z_pixel_um_network,
        cube_width=args.cube_width,
        cube_height=args.cube_height,
        cube_depth=args.cube_depth,
    )

    model = get_model(
        existing_model=args.trained_model,
        model_weights=args.model_weights,
        network_depth=models[args.network_depth],
        inference=True,
    )

    logging.info("Running inference")
    predictions = model.predict(
        inference_generator,
        use_multiprocessing=True,
        workers=workers,
        verbose=True,
    )
    predictions = predictions.round()
    predictions = predictions.astype("uint16")

    predictions = np.argmax(predictions, axis=1)
    cells_list = []
    non_cells_list = []

    # only go through the "extractable" cells
    for idx, cell in enumerate(inference_generator.ordered_cells):
        cell.type = predictions[idx] + 1
        if cell.type == 2:
            cells_list.append(cell)
        else:
            non_cells_list.append(cell)
    if args.prox_dist is not None:
        logging.info("Running proximity filtering")
        filt_cell_coords = proximity_filter(cells_list, args.prox_dist, [args.z_pixel_um, args.y_pixel_um, args.x_pixel_um])
        final_cells_list = []
        for cell_coord in filt_cell_coords:
            pos = {
                "x": cell_coord[2],

                "y": cell_coord[1],
                "z": cell_coord[0]
            }
            cell = Cell(pos, "cell")
            final_cells_list.append(cell)
    else:
        final_cells_list = cells_list
    msg = "Removed " + str(np.round((1 - (len(final_cells_list) / len(cells_list))) * 100)) + " % of rois marked as cells"
    logging.info(msg)
    result_list = final_cells_list
    result_list.extend(non_cells_list)
    logging.info("Saving classified cells")
    save_cells(
        result_list, args.paths.classification_out_file, save_csv=args.save_csv
    )
