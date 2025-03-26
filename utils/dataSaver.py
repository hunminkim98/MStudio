import os
import numpy as np
import pandas as pd
from tkinter import messagebox, filedialog

def save_to_trc(file_path, data, fps, marker_names, num_frames):
    """
    Save data to a TRC file.
    
    Args:
        file_path (str): Path to save the TRC file
        data (pd.DataFrame): DataFrame containing the marker data
        fps (float): Frames per second
        marker_names (list): List of marker names
        num_frames (int): Number of frames
    """
    header_lines = [
        "PathFileType\t4\t(X/Y/Z)\t{}\n".format(os.path.basename(file_path)),
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        "{}\t{}\t{}\t{}\tm\t{}\t{}\t{}\n".format(
            fps, fps, num_frames, len(marker_names), fps, 1, num_frames
        ),
        "\t".join(['Frame#', 'Time'] + [name + '\t\t' for name in marker_names]) + "\n",
        "\t".join(['', ''] + ['X\tY\tZ' for _ in marker_names]) + "\n"
    ]

    with open(file_path, 'w') as f:
        f.writelines(header_lines)
        data.to_csv(f, sep='\t', index=False, header=False, lineterminator='\n')

    messagebox.showinfo("Save Successful", f"Data saved to {file_path}")

def save_to_c3d(file_path, data, fps, marker_names, num_frames):
    """
    Save data to a C3D file.
    
    Args:
        file_path (str): Path to save the C3D file
        data (pd.DataFrame): DataFrame containing the marker data
        fps (float): Frames per second
        marker_names (list): List of marker names
        num_frames (int): Number of frames
    """
    try:
        import c3d
    except ImportError:
        messagebox.showerror("c3d Library Missing", "Please install the 'c3d' library to save in C3D format.")
        return

    try:
        writer = c3d.Writer(point_rate=float(fps), analog_rate=0)
        writer.set_point_labels(marker_names)

        all_frames = []
        
        for frame_idx in range(num_frames):
            points = np.zeros((len(marker_names), 5))
                
            for i, marker in enumerate(marker_names):
                try:
                    x = data.loc[frame_idx, f'{marker}_X'] * 1000.0  # Convert to mm
                    y = data.loc[frame_idx, f'{marker}_Y'] * 1000.0
                    z = data.loc[frame_idx, f'{marker}_Z'] * 1000.0

                    if np.isnan(x) or np.isnan(y) or np.isnan(z):
                        points[i, :3] = [0.0, 0.0, 0.0]
                        points[i, 3] = -1.0  # Residual
                        points[i, 4] = 0     # Camera_Mask
                    else:
                        points[i, :3] = [x, y, z]
                        points[i, 3] = 0.0   # Residual
                        points[i, 4] = 0     # Camera_Mask
                except Exception as e:
                    print(f"Error processing marker {marker} at frame {frame_idx}: {e}")
                    points[i, :3] = [0.0, 0.0, 0.0]
                    points[i, 3] = -1.0  # Residual
                    points[i, 4] = 0     # Camera_Mask

            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}...")

            all_frames.append((points, np.empty((0, 0))))

        writer.add_frames(all_frames)

        with open(file_path, 'wb') as h:
            writer.write(h)

        messagebox.showinfo("Save Successful", f"Data saved to {file_path}")

    except Exception as e:
        messagebox.showerror("Save Error", f"An error occurred while saving: {str(e)}\n\nPlease check the console for more details.")
        print(f"Detailed error: {e}")

def save_as(viewer):
    """
    Shows a save dialog and saves the current data to a TRC or C3D file
    
    Args:
        viewer: The TRCViewer instance containing the data to save
    
    Returns:
        bool: True if file was successfully saved, False otherwise
    """
    from tkinter import messagebox, filedialog
    import os
    
    if viewer.data is None:
        messagebox.showinfo("No Data", "There is no data to save.")
        return False

    file_path = filedialog.asksaveasfilename(
        defaultextension=".trc",
        filetypes=[("TRC files", "*.trc"), ("C3D files", "*.c3d"), ("All files", "*.*")]
    )

    if not file_path:
        return False

    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.trc':
            save_to_trc(file_path, viewer.data, viewer.fps_var.get(), viewer.marker_names, viewer.num_frames)
        elif file_extension == '.c3d':
            save_to_c3d(file_path, viewer.data, viewer.fps_var.get(), viewer.marker_names, viewer.num_frames)
        else:
            messagebox.showerror("Unsupported Format", "Unsupported file format.")
            return False
        return True
    except Exception as e:
        messagebox.showerror("Save Error", f"An error occurred while saving: {e}")
        return False
