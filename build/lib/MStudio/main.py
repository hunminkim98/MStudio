import customtkinter as ctk
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
