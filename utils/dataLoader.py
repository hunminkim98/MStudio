import pandas as pd
import c3d

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

def open_file(viewer):
    """
    Opens a motion file (TRC or C3D) and loads it into the viewer
    
    Args:
        viewer: The TRCViewer instance that will display the loaded data
    
    Returns:
        bool: True if file was successfully loaded, False otherwise
    """
    from tkinter import filedialog, messagebox
    import os
    
    file_path = filedialog.askopenfilename(
        filetypes=[("Motion files", "*.trc;*.c3d"), ("TRC files", "*.trc"), ("C3D files", "*.c3d"), ("All files", "*.*")]
    )

    if file_path:
        try:
            viewer.clear_current_state()

            viewer.current_file = file_path
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_path)[1].lower()
            viewer.title_label.configure(text=file_name)

            if file_extension == '.trc':
                header_lines, viewer.data, viewer.marker_names, frame_rate = read_data_from_trc(file_path)
            elif file_extension == '.c3d':
                header_lines, viewer.data, viewer.marker_names, frame_rate = read_data_from_c3d(file_path)
            else:
                raise Exception("Unsupported file format")

            viewer.num_frames = viewer.data.shape[0]
            viewer.original_data = viewer.data.copy(deep=True)
            viewer.calculate_data_limits()

            viewer.fps_var.set(str(int(frame_rate)))
            viewer.update_fps_label()

            # frame_slider related code
            viewer.frame_idx = 0
            viewer.update_timeline()

            viewer.current_model = viewer.available_models[viewer.model_var.get()]
            viewer.update_skeleton_pairs()
            viewer.detect_outliers()
            
            # 추가된 UI 업데이트 코드
            viewer.create_plot()
            viewer.reset_main_view()
            viewer.update_plot()

            if hasattr(viewer, 'canvas'):
                viewer.canvas.draw()
                viewer.canvas.flush_events()

            viewer.play_pause_button.configure(state='normal')
            viewer.loop_checkbox.configure(state='normal')

            viewer.is_playing = False
            viewer.play_pause_button.configure(text="▶")
            viewer.stop_button.configure(state='disabled')
            
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {str(e)}")
            return False
    
    return False
