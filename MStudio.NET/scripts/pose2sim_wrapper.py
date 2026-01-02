#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pose2Sim OpenSim Wrapper for MStudio
====================================
This script provides a command-line interface to call Pose2Sim's OpenSim functions
(scaling, inverse kinematics, body kinematics, GRF estimation) from external applications.

Usage:
    python pose2sim_wrapper.py scale --trc <path> --output <dir> --height <m> --mass <kg> --model <pose_model>
    python pose2sim_wrapper.py ik --trc <path> --output <dir> --model <pose_model>
    python pose2sim_wrapper.py bodykin --mot <path> --osim <path> --output <csv_path>
    python pose2sim_wrapper.py estimate_grf --com_csv <path> --mass <kg> --output <json_path>
    python pose2sim_wrapper.py check
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# Pose2Sim imports
try:
    from Pose2Sim.kinematics import (
        get_opensim_setup_dir,
        get_model_path,
        get_markers_path,
        get_scaling_setup,
        get_IK_Setup,
        perform_scaling,
        perform_IK
    )
    from Pose2Sim.common import read_trc
    import opensim
    POSE2SIM_AVAILABLE = True
except ImportError as e:
    POSE2SIM_AVAILABLE = False
    IMPORT_ERROR = str(e)


def run_body_kinematics(mot_path: str, osim_path: str, output_csv: str, direction: str = 'yup'):
    """
    Run BodyKinematics analysis to calculate Center of Mass (CoM) and body positions.
    Uses OpenSim's model.calcMassCenterPosition() for accurate whole-body CoM.
    
    Based on: bodykin_from_mot_osim.py by David Pagnon
    
    Args:
        mot_path: Path to .mot motion file (IK output)
        osim_path: Path to scaled .osim model file
        output_csv: Path to output CSV file
        direction: 'yup' (OpenSim default) or 'zup' (Blender)
    
    Returns:
        CSV file with columns: times, COM_x, COM_y, COM_z, body1_x, body1_y, ...
    """
    if not POSE2SIM_AVAILABLE:
        print(json.dumps({"success": False, "error": f"Pose2Sim not available: {IMPORT_ERROR}"}))
        return
    
    try:
        # Read model and motion files
        model = opensim.Model(osim_path)
        motion_data = opensim.TimeSeriesTable(mot_path)
        
        # Model: get model coordinates and bodies
        model_coordSet = model.getCoordinateSet()
        coordinateNames = motion_data.getColumnLabels()
        model_bodySet = model.getBodySet()
        bodies = [model_bodySet.get(i) for i in range(model_bodySet.getSize())]
        bodyNames = [b.getName() for b in bodies]
        
        # Motion: read coordinates and convert to radians
        times = motion_data.getIndependentColumn()
        motion_data_np = motion_data.getMatrix().to_numpy()
        
        for i, c in enumerate(coordinateNames):
            if model_coordSet.get(c).getMotionType() == 1:  # 1: rotation
                if motion_data.getTableMetaDataAsString('inDegrees') == 'yes':
                    motion_data_np[:, i] = motion_data_np[:, i] * np.pi / 180
        
        # Animate model and calculate CoM
        state = model.initSystem()
        loc_rot_frame_all = []
        com_frame_all = []
        H_zup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        
        for n in range(motion_data.getNumRows()):
            # Set model state for each time frame
            for c, coord in enumerate(coordinateNames):
                try:
                    model.getCoordinateSet().get(coord).setValue(state, motion_data_np[n, c], enforceContraints=False)
                except:
                    pass
            
            model.realizePosition(state)
            
            # Calculate Center of Mass (CoM) position using OpenSim's built-in method
            com_vec3 = model.calcMassCenterPosition(state)
            com_x, com_y, com_z = com_vec3.to_numpy()
            
            # Adjust CoM for y-up to z-up if needed
            if direction == 'zup':
                com_x_new, com_y_new, com_z_new, _ = H_zup @ np.array([com_x, com_y, com_z, 1])
                com_x, com_y, com_z = com_x_new, com_y_new, com_z_new
            
            com_frame_all.append([com_x, com_y, com_z])
            
            # Get body coordinates in ground
            loc_rot_frame = []
            for b in bodies:
                H_swig = b.getTransformInGround(state)
                T = H_swig.T().to_numpy()
                R_swig = H_swig.R()
                R = np.array([
                    [R_swig.get(0, 0), R_swig.get(0, 1), R_swig.get(0, 2)],
                    [R_swig.get(1, 0), R_swig.get(1, 1), R_swig.get(1, 2)],
                    [R_swig.get(2, 0), R_swig.get(2, 1), R_swig.get(2, 2)]
                ])
                H = np.block([[R, T.reshape(3, 1)], [np.zeros(3), 1]])
                
                if direction == 'zup':
                    H = H_zup @ H
                
                loc_x, loc_y, loc_z = H[0:3, 3]
                R_mat = H[0:3, 0:3]
                sy = np.sqrt(R_mat[1, 0]**2 + R_mat[0, 0]**2)
                if sy > 1e-6:
                    rot_x = np.arctan2(R_mat[2, 1], R_mat[2, 2])
                    rot_y = np.arctan2(-R_mat[2, 0], sy)
                    rot_z = np.arctan2(R_mat[1, 0], R_mat[0, 0])
                else:
                    rot_x = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
                    rot_y = np.arctan2(-R[2, 0], sy)
                    rot_z = 0
                loc_rot_frame.extend([loc_x, loc_y, loc_z, rot_x, rot_y, rot_z])
            
            loc_rot_frame_all.append(loc_rot_frame)
        
        # Export to CSV
        loc_rot_frame_all_np = np.array(loc_rot_frame_all)
        com_frame_all_np = np.array(com_frame_all)
        times_np = np.array(times).reshape(-1, 1)
        
        data_to_save = np.hstack((times_np, com_frame_all_np, loc_rot_frame_all_np))
        
        comHeader = 'COM_x, COM_y, COM_z, '
        bodyHeader = ''.join([f'{b}_x, {b}_y, {b}_z, {b}_rotx, {b}_roty, {b}_rotz, ' for b in bodyNames])[:-2]
        fullHeader = 'times, ' + comHeader + bodyHeader
        
        np.savetxt(output_csv, data_to_save, delimiter=',', header=fullHeader)
        
        print(json.dumps({
            "success": True,
            "output_csv": output_csv,
            "total_frames": len(times),
            "frame_rate": 1.0 / (times[1] - times[0]) if len(times) > 1 else 0
        }))
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


def estimate_grf_from_com(com_csv_path: str, mass_kg: float, output_json: str):
    """
    Estimate Ground Reaction Force from CoM data (from BodyKinematics output).
    
    GRF_y = m * a_y + m * g
    
    Based on: "Estimation of Ground Reaction Forces from Markerless Kinematics" (Colyer et al., 2023)
    
    Args:
        com_csv_path: Path to BodyKinematics CSV output (contains COM_x, COM_y, COM_z)
        mass_kg: Subject body mass in kg
        output_json: Path to save JSON result
    """
    try:
        g = 9.81  # m/s^2
        
        # Read header to determine column indices and separator
        with open(com_csv_path, 'r') as f:
            header_line = f.readline().strip()
        
        # Detect separator (tab or comma)
        separator = '\t' if '\t' in header_line else ','
        
        # Parse headers (remove # prefix and whitespace)
        headers = [h.strip().lstrip('#').strip().lower() for h in header_line.split(separator)]
        
        # Find column indices
        time_col = headers.index('times') if 'times' in headers else headers.index('time')
        com_y_col = headers.index('com_y')
        
        # Read CSV file with correct delimiter
        data = np.genfromtxt(com_csv_path, delimiter=separator, skip_header=1)
        
        # Extract time and CoM Y position (vertical)
        times = data[:, time_col]
        com_y = data[:, com_y_col]
        
        # Calculate frame rate
        dt = times[1] - times[0] if len(times) > 1 else 0.01
        frame_rate = 1.0 / dt
        
        # Filter: Low-pass Butterworth filter (12 Hz cutoff as per Colyer et al. 2023)
        from scipy import signal
        nyquist = frame_rate / 2
        cutoff = 12.0
        if cutoff < nyquist:
            order = 4
            b, a = signal.butter(order, cutoff / nyquist, btype='low')
            com_y_filtered = signal.filtfilt(b, a, com_y)
        else:
            com_y_filtered = com_y
        
        # Double differentiation: position -> velocity -> acceleration
        velocity = np.gradient(com_y_filtered, dt)
        acceleration = np.gradient(velocity, dt)
        
        # Calculate GRF: GRF = m * a + m * g (vertical direction, positive up)
        grf_vertical = mass_kg * acceleration + mass_kg * g
        
        # Find takeoff (when GRF drops to 0 or below body weight significantly)
        body_weight = mass_kg * g
        takeoff_idx = None
        for i in range(len(grf_vertical) - 1):
            # Takeoff: GRF drops below 10% of body weight
            if grf_vertical[i] > body_weight * 0.1 and grf_vertical[i + 1] <= body_weight * 0.1:
                takeoff_idx = i
                break
        
        if takeoff_idx is None:
            takeoff_idx = len(grf_vertical) - 1
        
        # Find landing (after takeoff, when GRF rises above body weight)
        landing_idx = None
        for i in range(takeoff_idx + 5, len(grf_vertical)):
            if grf_vertical[i] > body_weight * 0.5:
                landing_idx = i
                break
        
        # Peak GRF (during propulsion, before takeoff)
        peak_grf = float(np.max(grf_vertical[:takeoff_idx]))
        peak_grf_frame = int(np.argmax(grf_vertical[:takeoff_idx]))
        
        # Net Impulse (integral of force - body weight, from start to takeoff)
        # This represents the net impulse that causes the change in momentum
        net_impulse = float(np.trapz(grf_vertical[:takeoff_idx] - body_weight, dx=dt))
        
        # RFD (Rate of Force Development) - slope from min to peak
        # Find the countermovement phase (minimum GRF before peak)
        min_idx = int(np.argmin(grf_vertical[:peak_grf_frame])) if peak_grf_frame > 0 else 0
        
        if peak_grf_frame > min_idx:
            rfd = (grf_vertical[peak_grf_frame] - grf_vertical[min_idx]) / ((peak_grf_frame - min_idx) * dt)
        else:
            rfd = 0.0
        
        # Prepare output
        result = {
            "success": True,
            "com_csv_path": com_csv_path,
            "mass_kg": mass_kg,
            "frame_rate": float(frame_rate),
            "total_frames": len(grf_vertical),
            "metrics": {
                "peak_vertical_grf_N": peak_grf,
                "peak_grf_frame": peak_grf_frame,
                "net_vertical_impulse_Ns": net_impulse,
                "rfd_N_per_s": float(rfd),
                "takeoff_frame": int(takeoff_idx),
                "landing_frame": int(landing_idx) if landing_idx else None,
                "body_weight_N": float(body_weight)
            },
            "grf_timeseries": {
                "time_s": times.tolist(),
                "grf_vertical_N": grf_vertical.tolist()
            }
        }
        
        # Save to JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(json.dumps({"success": True, "output": output_json}))
        return result
        
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return None


def run_scaling(trc_path: str, output_dir: str, height: float, mass: float, 
                pose_model: str = "COCO_133", use_simple_model: bool = False):
    """
    Run OpenSim model scaling using Pose2Sim.
    """
    if not POSE2SIM_AVAILABLE:
        print(json.dumps({"success": False, "error": f"Pose2Sim not available: {IMPORT_ERROR}"}))
        return
    
    trc_file = Path(trc_path)
    kinematics_dir = Path(output_dir)
    kinematics_dir.mkdir(parents=True, exist_ok=True)
    osim_setup_dir = get_opensim_setup_dir()
    
    try:
        perform_scaling(
            trc_file=trc_file,
            pose_model=pose_model,
            kinematics_dir=kinematics_dir,
            osim_setup_dir=osim_setup_dir,
            use_simple_model=use_simple_model,
            right_left_symmetry=True,
            subject_height=height,
            subject_mass=mass,
            remove_scaling_setup=True
        )
        
        scaled_model_path = kinematics_dir / (trc_file.stem + '.osim')
        print(json.dumps({
            "success": True,
            "scaled_model": str(scaled_model_path)
        }))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


def run_ik(trc_path: str, output_dir: str, pose_model: str = "COCO_133"):
    """
    Run OpenSim Inverse Kinematics using Pose2Sim.
    """
    if not POSE2SIM_AVAILABLE:
        print(json.dumps({"success": False, "error": f"Pose2Sim not available: {IMPORT_ERROR}"}))
        return
    
    trc_file = Path(trc_path)
    kinematics_dir = Path(output_dir)
    osim_setup_dir = get_opensim_setup_dir()
    
    try:
        perform_IK(
            trc_file=trc_file,
            kinematics_dir=kinematics_dir,
            osim_setup_dir=osim_setup_dir,
            pose_model=pose_model,
            remove_IK_setup=True
        )
        
        mot_path = kinematics_dir / (trc_file.stem + '.mot')
        print(json.dumps({
            "success": True,
            "motion_file": str(mot_path)
        }))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


def main():
    parser = argparse.ArgumentParser(description="Pose2Sim OpenSim Wrapper for MStudio")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scale command
    scale_parser = subparsers.add_parser('scale', help='Scale OpenSim model')
    scale_parser.add_argument('--trc', required=True, help='Path to TRC file')
    scale_parser.add_argument('--output', required=True, help='Output directory')
    scale_parser.add_argument('--height', type=float, required=True, help='Subject height (m)')
    scale_parser.add_argument('--mass', type=float, required=True, help='Subject mass (kg)')
    scale_parser.add_argument('--model', default='COCO_133', help='Pose model name')
    scale_parser.add_argument('--simple', action='store_true', help='Use simple model')
    
    # IK command
    ik_parser = subparsers.add_parser('ik', help='Run Inverse Kinematics')
    ik_parser.add_argument('--trc', required=True, help='Path to TRC file')
    ik_parser.add_argument('--output', required=True, help='Output directory')
    ik_parser.add_argument('--model', default='COCO_133', help='Pose model name')
    
    # BodyKinematics command (NEW)
    bodykin_parser = subparsers.add_parser('bodykin', help='Run BodyKinematics analysis (CoM calculation)')
    bodykin_parser.add_argument('--mot', required=True, help='Path to .mot motion file')
    bodykin_parser.add_argument('--osim', required=True, help='Path to scaled .osim model file')
    bodykin_parser.add_argument('--output', required=True, help='Output CSV path')
    bodykin_parser.add_argument('--direction', default='yup', choices=['yup', 'zup'], help='Coordinate direction')
    
    # GRF estimation command (updated to use CoM CSV)
    grf_parser = subparsers.add_parser('estimate_grf', help='Estimate GRF from BodyKinematics CoM data')
    grf_parser.add_argument('--com_csv', required=True, help='Path to BodyKinematics CSV (with CoM data)')
    grf_parser.add_argument('--mass', type=float, required=True, help='Subject mass (kg)')
    grf_parser.add_argument('--output', required=True, help='Output JSON path')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check Pose2Sim availability')
    
    args = parser.parse_args()
    
    if args.command == 'scale':
        run_scaling(args.trc, args.output, args.height, args.mass, args.model, args.simple)
    elif args.command == 'ik':
        run_ik(args.trc, args.output, args.model)
    elif args.command == 'bodykin':
        run_body_kinematics(args.mot, args.osim, args.output, args.direction)
    elif args.command == 'estimate_grf':
        estimate_grf_from_com(args.com_csv, args.mass, args.output)
    elif args.command == 'check':
        if POSE2SIM_AVAILABLE:
            print(json.dumps({"available": True, "version": "0.10.0"}))
        else:
            print(json.dumps({"available": False, "error": IMPORT_ERROR}))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
