import customtkinter as ctk
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MStudio.app import TRCViewer

## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
# from importlib.metadata import version
# __version__ = version('MStudio')
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"


def main():
    """
    Main function to run the MarkerStudio application
    """
    # Theme
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = TRCViewer()
    app.mainloop()

if __name__ == "__main__":
    main()


# TODO:
# 1. Add skeleton lines for trunk.
# 2. Drag and select multiple markers. <- Should change the logic of the left click.
# 3. Choice the view by clicking the plane (inspired from OpenSim GUI)
# 4. Marker size, color, opacity.
# 5. Add an arc for visualizing the range of motion.

