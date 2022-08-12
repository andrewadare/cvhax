from pathlib import Path
import sys
from typing import Callable, Optional

import cv2


def save(im, output_dir: Path):
    """Save an image that is named using an internally-incremented counter."""
    if not hasattr(save, "counter"):
        save.counter = 0  # type: ignore
    fname = f"{save.counter:05d}.jpg"  # type: ignore
    cv2.imwrite(str(output_dir / fname), im)
    print("Saved", fname)
    save.counter += 1  # type: ignore


def display(
    im,
    window_name: str = "win",
    delay_ms: int = 1,
    on_quit: Callable[[], None] = None,
    save_dir: Optional[Path] = None,
):
    """Display `im` in named window, pausing for `delay_ms` ms or until the next
    keypress. Note that `delay_ms=0` blocks indefinitely.

    Pressing "q" closes the window and stops the program. "s" saves a snapshot
    named 00000.jpg, 00001.jpg, etc.

    Parameters
    ----------
    im : OpenCV image (numpy.ndarray)
        Image to be displayed
    window_name : string (optional)
        Name of GUI window. Appears in title bar.
    delay_ms : int
        Number of milliseconds to pause before execution continues.
    on_quit : function
        Callback to be executed before termination of program. For example,
        may contain logic to save in-memory data to disk before closing.
    save_dir: if provided, and 's' is typed, save frame there.
    """
    cv2.imshow(window_name, im)

    char = cv2.waitKey(delay_ms)
    if char == ord("s") and save_dir is not None:
        save(im, save_dir)
    if char == ord("q"):
        if callable(on_quit):
            on_quit()
        cv2.destroyAllWindows()
        sys.exit(0)
