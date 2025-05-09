import pandas as pd
import c3d
import logging
import os
import json
import re
import numpy as np

logger = logging.getLogger(__name__)

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

def read_data_from_c3d(c3d_file_path):
    """
    Read data from a C3D file and return header lines, data frame, marker names, and frame rate.
    """
    try:
        with open(c3d_file_path, 'rb') as f:
            reader = c3d.Reader(f)
            point_labels = reader.point_labels
            frame_rate = reader.header.frame_rate
            first_frame = reader.header.first_frame
            last_frame = reader.header.last_frame

            point_labels = [label.strip() for label in point_labels if label.strip()]
            point_labels = list(dict.fromkeys(point_labels))

            frames = []
            times = []
            marker_data = {label: {'X': [], 'Y': [], 'Z': []} for label in point_labels}

            for i, points, analog in reader.read_frames():
                frames.append(i)
                times.append(i / frame_rate)
                points_meters = points[:, :3] / 1000.0

                for j, label in enumerate(point_labels):
                    if j < len(points_meters):
                        marker_data[label]['X'].append(points_meters[j, 0])
                        marker_data[label]['Y'].append(points_meters[j, 1])
                        marker_data[label]['Z'].append(points_meters[j, 2])

            data_dict = {'Frame#': frames, 'Time': times}

            for label in point_labels:
                if label in marker_data:
                    data_dict[f'{label}_X'] = marker_data[label]['X']
                    data_dict[f'{label}_Y'] = marker_data[label]['Y']
                    data_dict[f'{label}_Z'] = marker_data[label]['Z']

            data = pd.DataFrame(data_dict)

            header_lines = [
                f"PathFileType\t4\t(X/Y/Z)\t{c3d_file_path}\n",
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
                f"{frame_rate}\t{frame_rate}\t{len(frames)}\t{len(point_labels)}\tm\t{frame_rate}\t{first_frame}\t{last_frame}\n",
                "\t".join(['Frame#', 'Time'] + point_labels) + "\n",
                "\t".join(['', ''] + ['X\tY\tZ' for _ in point_labels]) + "\n"
            ]

            return header_lines, data, point_labels, frame_rate

    except Exception as e:
        raise Exception(f"Error reading C3D file: {str(e)}")

def read_data_from_trc(trc_file_path):
    """
    Read data from a TRC file and return header lines, data frame, marker names, and frame rate.
    """
    with open(trc_file_path, 'r') as f:
        lines = f.readlines()

    header_lines = lines[:5]

    try:
        frame_rate = float(header_lines[2].split('\t')[0])
    except (IndexError, ValueError):
        frame_rate = 30.0

    marker_names_line = lines[3].strip().split('\t')[2:]

    marker_names = []
    for name in marker_names_line:
        if name.strip() and name not in marker_names:
            marker_names.append(name.strip())

    column_names = ['Frame#', 'Time']
    for marker in marker_names:
        column_names.extend([f'{marker}_X', f'{marker}_Y', f'{marker}_Z'])

    data = pd.read_csv(trc_file_path, sep='\t', skiprows=6, names=column_names)

    return header_lines, data, marker_names, frame_rate

def read_data_from_json_folder(json_folder_path, coordinate_system='Y-up'):
    """
    Read data from a folder of JSON files (OpenPose format) and return header lines, data frame, marker names, and frame rate.
    
    Parameters:
    -----------
    json_folder_path : str
        Path to the folder containing JSON files
    coordinate_system : str, optional
        Coordinate system to use ('Y-up' or 'Z-up'). Default is 'Y-up'.
        
    Returns:
    --------
    header_lines : list
        List of header lines for the data
    data : pandas.DataFrame
        DataFrame containing the marker data
    marker_names : list
        List of marker names (generic keypoint indices)
    frame_rate : float
        Frame rate of the data
    """
    try:
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(json_folder_path) if f.lower().endswith('.json')]
        
        if not json_files:
            raise Exception("No JSON files found in the folder")
        
        # Sort files by frame number if possible
        def extract_frame_number(filename):
            match = re.search(r'(\d+)\.json$', filename)
            return int(match.group(1)) if match else 0
        
        json_files.sort(key=extract_frame_number)
        
        # Read the first file to get the structure
        with open(os.path.join(json_folder_path, json_files[0]), 'r') as f:
            first_json = json.load(f)
        
        # Check if the file has the expected structure
        if 'people' not in first_json or not first_json['people']:
            raise Exception("Invalid JSON format: 'people' array not found or empty")
        
        # Get the number of keypoints from the first person in the first file
        num_keypoints = 0
        if first_json['people']:
            pose_keypoints = first_json['people'][0].get('pose_keypoints_2d', [])
            num_keypoints = len(pose_keypoints) // 3  # Each keypoint has x, y, confidence
        
        # Generate generic keypoint names based on indices
        # These will be updated later when a skeleton is selected in the GUI
        keypoint_names = [f"Keypoint_{i}" for i in range(num_keypoints)]
        
        # Initialize data structures
        frames = []
        times = []
        frame_rate = 30.0  # Default frame rate, can be adjusted
        
        # Initialize marker data dictionary
        marker_data = {name: {'X': [], 'Y': [], 'Z': []} for name in keypoint_names}
        
        # Process each JSON file
        for i, json_file in enumerate(json_files):
            file_path = os.path.join(json_folder_path, json_file)
            
            with open(file_path, 'r') as f:
                data_json = json.load(f)
            
            frames.append(i)
            times.append(i / frame_rate)
            
            # If there are people in the JSON
            if data_json['people']:
                # Get the first person's keypoints
                person = data_json['people'][0]
                pose_keypoints = person.get('pose_keypoints_2d', [])
                
                # Process each keypoint
                for j, name in enumerate(keypoint_names):
                    if j * 3 + 2 < len(pose_keypoints):
                        x = pose_keypoints[j * 3]
                        y = pose_keypoints[j * 3 + 1]
                        confidence = pose_keypoints[j * 3 + 2]
                        
                        # Only use keypoints with confidence above threshold
                        if confidence > 0.0:
                            marker_data[name]['X'].append(x / 1000.0)  # Convert to meters
                            
                            marker_data[name]['Y'].append(-y / 1000.0)  # Convert to meters and flip Y
                            
                            # Handle Z value based on coordinate system
                            if coordinate_system == 'Y-up':
                                marker_data[name]['Z'].append(0.0)  # Z is 0 for Y-up
                            else:  # Z-up
                                # For Z-up, we swap Y and Z (Y becomes 0, Z takes the Y value)
                                y_value = -y / 1000.0  # Store Y value before overwriting (already flipped)
                                marker_data[name]['Z'].append(y_value)  # Z takes the Y value
                                marker_data[name]['Y'][-1] = 0.0  # Y becomes 0
                        else:
                            marker_data[name]['X'].append(np.nan)
                            marker_data[name]['Y'].append(np.nan)
                            marker_data[name]['Z'].append(np.nan)
                    else:
                        marker_data[name]['X'].append(np.nan)
                        marker_data[name]['Y'].append(np.nan)
                        marker_data[name]['Z'].append(np.nan)
            else:
                # If no people detected, add NaN values
                for name in keypoint_names:
                    marker_data[name]['X'].append(np.nan)
                    marker_data[name]['Y'].append(np.nan)
                    marker_data[name]['Z'].append(np.nan)
        
        # Create the data dictionary
        data_dict = {'Frame#': frames, 'Time': times}
        
        for name in keypoint_names:
            data_dict[f'{name}_X'] = marker_data[name]['X']
            data_dict[f'{name}_Y'] = marker_data[name]['Y']
            data_dict[f'{name}_Z'] = marker_data[name]['Z']
        
        # Create the DataFrame
        data = pd.DataFrame(data_dict)
        
        # Center the character at the origin (based on first frame)
        if not data.empty:
            # Find the first frame with valid data
            first_valid_frame = 0
            while first_valid_frame < len(frames):
                # Check if this frame has enough valid data
                valid_markers = 0
                for name in keypoint_names:
                    if not np.isnan(data.loc[first_valid_frame, f'{name}_X']):
                        valid_markers += 1
                
                if valid_markers >= 3:  # At least 3 valid markers to determine position
                    break
                first_valid_frame += 1
            
            if first_valid_frame < len(frames):
                # Calculate the centroid of the character in the first valid frame
                valid_x = []
                valid_y = []
                valid_z = []
                
                for name in keypoint_names:
                    x_val = data.loc[first_valid_frame, f'{name}_X']
                    y_val = data.loc[first_valid_frame, f'{name}_Y']
                    z_val = data.loc[first_valid_frame, f'{name}_Z']
                    
                    if not np.isnan(x_val) and not np.isnan(y_val) and not np.isnan(z_val):
                        valid_x.append(x_val)
                        valid_y.append(y_val)
                        valid_z.append(z_val)
                
                if valid_x and valid_y and valid_z:
                    # Calculate centroid
                    centroid_x = np.mean(valid_x)
                    centroid_y = np.mean(valid_y)
                    centroid_z = np.mean(valid_z)
                    
                    # Find the minimum Y value (lowest point, feet)
                    min_y = min(valid_y)
                    
                    # Calculate the Y offset to position the lowest point at origin
                    # This is the distance from the centroid to the lowest point
                    y_offset = centroid_y - min_y
                    
                    # Store original data
                    original_data = data.copy(deep=True)
                    
                    # Translate all frames to position the lowest point at origin
                    for i in range(len(frames)):
                        for name in keypoint_names:
                            x_col = f'{name}_X'
                            y_col = f'{name}_Y'
                            z_col = f'{name}_Z'
                            
                            if not np.isnan(data.loc[i, x_col]):
                                data.loc[i, x_col] -= centroid_x
                            
                            if not np.isnan(data.loc[i, y_col]):
                                # Apply centroid offset plus the additional offset to position lowest point at origin
                                data.loc[i, y_col] -= (centroid_y - y_offset)
                            
                            if not np.isnan(data.loc[i, z_col]):
                                data.loc[i, z_col] -= centroid_z
        
        # Create header lines similar to TRC format
        header_lines = [
            f"PathFileType\t4\t(X/Y/Z)\t{json_folder_path}\n",
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
            f"{frame_rate}\t{frame_rate}\t{len(frames)}\t{len(keypoint_names)}\tm\t{frame_rate}\t0\t{len(frames)}\n",
            "\t".join(['Frame#', 'Time'] + keypoint_names) + "\n",
            "\t".join(['', ''] + ['X\tY\tZ' for _ in keypoint_names]) + "\n"
        ]
        
        return header_lines, data, keypoint_names, frame_rate
    
    except Exception as e:
        raise Exception(f"Error reading JSON files: {str(e)}")

def open_file(viewer):
    """
    Opens motion files (TRC, C3D, or JSON) and loads them into the viewer.
    Multiple JSON files can be selected at once.
    """
    from tkinter import filedialog, messagebox
    import os
    
    # Open file dialog with support for multiple selection
    file_paths = filedialog.askopenfilenames(
        filetypes=[
            ("All supported files", "*.trc;*.c3d;*.json"),
            ("TRC files", "*.trc"), 
            ("C3D files", "*.c3d"), 
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ],
        title="Select motion file(s)"
    )

    if not file_paths:
        return False
        
    # Convert tuple to list
    file_paths = list(file_paths)
    
    # Check if we have JSON files
    json_files = [f for f in file_paths if os.path.splitext(f)[1].lower() == '.json']
    non_json_files = [f for f in file_paths if os.path.splitext(f)[1].lower() != '.json']
    
    # Validate selection - can't mix JSON with other file types
    if json_files and non_json_files:
        messagebox.showerror("Error", "Cannot mix JSON files with other file types. Please select either JSON files only or a single TRC/C3D file.")
        return False
    
    # Validate selection - can only select one non-JSON file
    if len(non_json_files) > 1:
        messagebox.showerror("Error", "Please select only one TRC or C3D file at a time.")
        return False
    
    try:
        # Reset the current state
        viewer.clear_current_state()
        
        # Process JSON files
        if json_files:
            # Create a temporary directory to store JSON file paths
            import tempfile
            import shutil
            
            # Get the directory of the first JSON file to use as a base name
            base_dir = os.path.dirname(json_files[0])
            base_name = os.path.basename(base_dir)
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp(prefix="mstudio_json_")
            
            # Copy all selected JSON files to the temporary directory
            for json_file in json_files:
                dest_file = os.path.join(temp_dir, os.path.basename(json_file))
                shutil.copy2(json_file, dest_file)
            
            # Set the file information
            viewer.current_file = temp_dir
            viewer.title_label.configure(text=f"JSON Files: {len(json_files)} files")
            
            # Load the data from the JSON folder
            coordinate_system = 'Z-up' if viewer.is_z_up else 'Y-up'
            header_lines, viewer.data, viewer.marker_names, frame_rate = read_data_from_json_folder(temp_dir, coordinate_system)
        
        # Process TRC or C3D file
        else:
            file_path = non_json_files[0]  # We've validated there's only one
            
            # Set the file information
            viewer.current_file = file_path
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            viewer.title_label.configure(text=file_name)
            
            # Load the data based on the file extension
            if file_extension == '.trc':
                header_lines, viewer.data, viewer.marker_names, frame_rate = read_data_from_trc(file_path)
            elif file_extension == '.c3d':
                header_lines, viewer.data, viewer.marker_names, frame_rate = read_data_from_c3d(file_path)
            else:
                raise Exception("Unsupported file format")
        
        # Reset the data
        viewer.num_frames = viewer.data.shape[0]
        viewer.original_data = viewer.data.copy(deep=True)
        viewer.calculate_data_limits()
        viewer.fps_var.set(str(int(frame_rate)))
        viewer.update_fps_label()
        viewer.frame_idx = 0
        viewer.update_timeline()
        
        # Set the skeleton model
        viewer.current_model = viewer.available_models[viewer.model_var.get()]
        viewer.update_skeleton_pairs()
        viewer.detect_outliers()
        
        # Create the plot (now always OpenGL)
        viewer.create_plot()
        
        # Wait for the renderer to initialize (if needed)
        viewer.update_idletasks() 
        
        # Reset the view and update safely
        try:
            viewer.reset_main_view()
        except Exception as e:
            print(f"Error resetting the view: {e}. Continuing...")
            
        # Update the plot (if error, remove the fallback logic)
        try:
            viewer.update_plot()
        except Exception as e:
            print(f"Error updating the plot: {e}")
        
        # Update the UI controls
        viewer.play_pause_button.configure(state='normal')
        viewer.loop_checkbox.configure(state='normal')
        viewer.is_playing = False
        viewer.play_pause_button.configure(text="▶")
        viewer.stop_button.configure(state='disabled')
        
        return True
                
    except Exception as e:
        import traceback
        logger.error("Error loading file(s): %s", e, exc_info=True)
        messagebox.showerror("Error", f"Failed to open file(s): {str(e)}")
        return False
