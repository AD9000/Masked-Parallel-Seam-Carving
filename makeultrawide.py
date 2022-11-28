import time

start = time.time()

from PIL import Image
import seam_carving
from multiprocessing import freeze_support
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor
from glob import glob
import time
import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser(
    prog="Masked Seam Carving (Anime ver)",
    description="Creating 21:9 2D art from 16:9 versions",
)
parser.add_argument(
    "src_folder", type=str, help="Folder for images to be extended"
)
parser.add_argument(
    "dest_folder", type=str, help="Destination Folder to save extended images"
)
parser.add_argument(
    "-p", "--process_count", type=int, default=4, required=False
)
parser.add_argument(
    "-t", "--thread_count", type=int, default=16, required=False
)


def extend(args):
    src, mask = args
    src_w, src_h = src.size
    new_w = int(21.0 / 16.0 * src_w)
    mask = mask.convert("L")

    dst = seam_carving.resize(
        src,
        (new_w, src_h),
        energy_mode="forward",  # Choose from {backward, forward}
        order="width-first",  # Choose from {width-first, height-first}
        keep_mask=mask,
    )
    return Image.fromarray(dst)


def load_image(image_path):
    if not Path(image_path).exists():
        raise Exception(f"Image {image_path} does not exists")

    return Image.open(image_path).convert("RGB")


def load_image_cv2(image_path):
    if not Path(image_path).exists():
        raise Exception(f"Image {image_path} does not exists")

    return cv2.imread(image_path)


def main(src_folder, dest_folder, proc_count=4, thread_count=16):
    start = time.time()
    src_folder, dest_folder = Path(src_folder), Path(dest_folder)

    if not src_folder.exists():
        print(f"Source folder, {src_folder} does not exist")
        exit(0)

    # creating destination folder if it does not exist
    dest_folder.mkdir(exist_ok=True)

    # optimizing imports
    from mask import Masking

    predictor = Masking()

    print("Loading Images...")
    cv2_images = []
    raw_images = []
    image_paths = glob(str(src_folder.joinpath("*")))
    with ThreadPoolExecutor(thread_count) as pool:
        raw_images = list(
            pool.map(
                load_image,
                image_paths,
            )
        )

        cv2_images = list(pool.map(load_image_cv2, image_paths))

    # images = list(zip(raw_images, image_paths))
    print("Images Loaded")

    print("Generating Masks...")
    # mask generation
    masks = predictor.predict_masks(cv2_images)
    print("Masks generated")

    print("Extending Images...")
    with ProcessPoolExecutor(proc_count) as pool:
        results = list(pool.map(extend, zip(raw_images, masks)))
    print("Images Extended")

    print("Saving Images")
    with ThreadPoolExecutor(thread_count) as pool:
        list(
            pool.map(
                lambda x: x[0].save(dest_folder.joinpath(Path(x[1]).name)),
                zip(results, image_paths),
            )
        )
    print("Images Saved.")

    print("Done :)")
    print(f"Time taken: {time.time() - start}")


if __name__ == "__main__":
    freeze_support()

    args = parser.parse_args()
    main(
        args.src_folder,
        args.dest_folder,
        proc_count=args.process_count,
        thread_count=args.thread_count,
    )
    print(f"Done in {time.time() - start:.02f} seconds")
