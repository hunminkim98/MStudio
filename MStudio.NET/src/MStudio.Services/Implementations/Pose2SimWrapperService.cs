using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using MStudio.Core.Interfaces;
using MStudio.Core.Models.Analysis;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// Service for calling Pose2Sim Python wrapper.
    /// Provides OpenSim integration for scaling, IK, BodyKinematics, and GRF estimation.
    /// </summary>
    public class Pose2SimWrapperService : IPose2SimService
    {
        private readonly string _pythonPath;
        private readonly string _wrapperScriptPath;

        /// <summary>
        /// Initializes the Pose2Sim wrapper service.
        /// </summary>
        /// <param name="pythonPath">Path to Python executable. Default: "python"</param>
        /// <param name="wrapperScriptPath">Path to pose2sim_wrapper.py script. If null, uses default location.</param>
        public Pose2SimWrapperService(string pythonPath = "python", string? wrapperScriptPath = null)
        {
            _pythonPath = pythonPath;
            _wrapperScriptPath = wrapperScriptPath 
                ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "scripts", "pose2sim_wrapper.py");
        }

        /// <summary>
        /// Creates a Pose2SimWrapperService from configuration file.
        /// Looks for opensim_config.json in the application base directory.
        /// </summary>
        public static Pose2SimWrapperService CreateFromConfig(string? configPath = null)
        {
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            
            // Try multiple config file locations
            var configPaths = new[]
            {
                configPath,
                Path.Combine(baseDir, "opensim_config.json"),
                Path.Combine(baseDir, "..", "..", "..", "..", "src", "MStudio.App", "opensim_config.json")
            };

            string? pythonPath = null;
            string? scriptPath = null;

            foreach (var path in configPaths)
            {
                if (!string.IsNullOrEmpty(path) && File.Exists(path))
                {
                    try
                    {
                        var json = File.ReadAllText(path);
                        using var doc = JsonDocument.Parse(json);
                        var root = doc.RootElement;
                        
                        pythonPath = root.TryGetProperty("PythonPath", out var pp) ? pp.GetString() : null;
                        var scriptPathRel = root.TryGetProperty("Pose2SimScriptPath", out var sp) ? sp.GetString() : null;
                        
                        // Resolve script path - try baseDir first (for deployed app)
                        if (!string.IsNullOrEmpty(scriptPathRel))
                        {
                            // Priority 1: Output directory (deployed)
                            var deployedPath = Path.GetFullPath(Path.Combine(baseDir, scriptPathRel));
                            if (File.Exists(deployedPath))
                            {
                                scriptPath = deployedPath;
                                break;
                            }
                            
                            // Priority 2: Development - relative to project root
                            var devPath = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", scriptPathRel));
                            if (File.Exists(devPath))
                            {
                                scriptPath = devPath;
                                break;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Failed to load config from {path}: {ex.Message}");
                    }
                }
            }
            
            // Fallback: try to find script in common locations
            if (scriptPath == null)
            {
                var scriptLocations = new[]
                {
                    Path.Combine(baseDir, "scripts", "pose2sim_wrapper.py"),
                    Path.Combine(baseDir, "..", "..", "..", "..", "scripts", "pose2sim_wrapper.py")
                };
                
                foreach (var loc in scriptLocations)
                {
                    var fullPath = Path.GetFullPath(loc);
                    if (File.Exists(fullPath))
                    {
                        scriptPath = fullPath;
                        break;
                    }
                }
            }
            
            return new Pose2SimWrapperService(pythonPath ?? "python", scriptPath);
        }

        /// <summary>
        /// Checks if Pose2Sim is available.
        /// </summary>
        public async Task<(bool Available, string? Version, string? Error)> CheckAvailabilityAsync()
        {
            var result = await RunCommandAsync("check");
            if (result.Success && result.Output != null)
            {
                using var doc = JsonDocument.Parse(result.Output);
                var root = doc.RootElement;
                if (root.TryGetProperty("available", out var avail) && avail.GetBoolean())
                {
                    var version = root.TryGetProperty("version", out var v) ? v.GetString() : null;
                    return (true, version, null);
                }
                else
                {
                    var error = root.TryGetProperty("error", out var e) ? e.GetString() : "Unknown error";
                    return (false, null, error);
                }
            }
            return (false, null, result.Error ?? "Failed to run check command");
        }

        /// <summary>
        /// Runs BodyKinematics analysis to calculate Center of Mass (CoM) positions.
        /// Uses OpenSim's model.calcMassCenterPosition() for accurate whole-body CoM.
        /// </summary>
        /// <param name="motPath">Path to .mot motion file (IK output)</param>
        /// <param name="osimPath">Path to scaled .osim model file</param>
        /// <param name="outputCsvPath">Path to output CSV file</param>
        /// <param name="direction">Coordinate direction: 'yup' (OpenSim) or 'zup' (Blender)</param>
        /// <returns>Success status, output CSV path, frame rate, and error message</returns>
        public async Task<(bool Success, string? OutputCsvPath, float FrameRate, string? Error)> RunBodyKinematicsAsync(
            string motPath, string osimPath, string outputCsvPath, string direction = "yup")
        {
            var args = $"bodykin --mot \"{motPath}\" --osim \"{osimPath}\" --output \"{outputCsvPath}\" --direction {direction}";
            var result = await RunCommandAsync(args);
            
            if (!result.Success)
            {
                return (false, null, 0, result.Error);
            }
            
            if (result.Output != null)
            {
                using var doc = JsonDocument.Parse(result.Output);
                var root = doc.RootElement;
                if (root.TryGetProperty("success", out var s) && s.GetBoolean())
                {
                    var csvPath = root.TryGetProperty("output_csv", out var cp) ? cp.GetString() : null;
                    var frameRate = root.TryGetProperty("frame_rate", out var fr) ? (float)fr.GetDouble() : 0;
                    return (true, csvPath, frameRate, null);
                }
                else
                {
                    var error = root.TryGetProperty("error", out var e) ? e.GetString() : "Unknown error";
                    return (false, null, 0, error);
                }
            }
            
            return (false, null, 0, "No output from bodykin command");
        }

        /// <summary>
        /// Estimates Ground Reaction Force from CoM data (from BodyKinematics output).
        /// Based on: "Estimation of Ground Reaction Forces from Markerless Kinematics" (Colyer et al., 2023)
        /// Formula: GRF_y = m * a_y + m * g
        /// </summary>
        /// <param name="comCsvPath">Path to BodyKinematics CSV output (contains COM_x, COM_y, COM_z)</param>
        /// <param name="massKg">Subject body mass in kg</param>
        /// <returns>GRF estimation result with peak force, impulse, RFD, and time series</returns>
        public async Task<GRFEstimationResult> EstimateGRFAsync(
            string comCsvPath, 
            float massKg, 
            int? takeoffFrame = null, 
            int? landingFrame = null,
            int? lowestCoMFrame = null)
        {
            var outputPath = Path.Combine(Path.GetTempPath(), $"grf_result_{Guid.NewGuid():N}.json");
            
            try
            {
                var args = $"estimate_grf --com_csv \"{comCsvPath}\" --mass {massKg} --output \"{outputPath}\"";
                
                if (takeoffFrame.HasValue) args += $" --takeoff {takeoffFrame.Value}";
                if (landingFrame.HasValue) args += $" --landing {landingFrame.Value}";
                if (lowestCoMFrame.HasValue) args += $" --lowest_com {lowestCoMFrame.Value}";
                
                var result = await RunCommandAsync(args);
                
                if (!result.Success)
                {
                    return new GRFEstimationResult { Success = false, Error = result.Error };
                }
                
                // Check if Python printed success directly
                if (result.Output != null && result.Output.Contains("\"success\": true"))
                {
                    // Read output JSON file
                    if (!File.Exists(outputPath))
                    {
                        return new GRFEstimationResult { Success = false, Error = "Output file not created" };
                    }
                    
                    var json = await File.ReadAllTextAsync(outputPath);
                    using var doc = JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    
                    var grfResult = new GRFEstimationResult
                    {
                        Success = root.GetProperty("success").GetBoolean(),
                        TrcPath = root.TryGetProperty("com_csv_path", out var tp) ? tp.GetString() : null,
                        MassKg = root.TryGetProperty("mass_kg", out var m) ? (float)m.GetDouble() : 0,
                        FrameRate = root.TryGetProperty("frame_rate", out var fr) ? (float)fr.GetDouble() : 0,
                        TotalFrames = root.TryGetProperty("total_frames", out var tf) ? tf.GetInt32() : 0
                    };
                    
                    // Parse metrics
                    if (root.TryGetProperty("metrics", out var metrics))
                    {
                        grfResult.Metrics = new GRFMetrics
                        {
                            PeakVerticalGrfN = metrics.TryGetProperty("peak_vertical_grf_N", out var peak) ? (float)peak.GetDouble() : 0,
                            NetVerticalImpulseNs = metrics.TryGetProperty("net_vertical_impulse_Ns", out var imp) ? (float)imp.GetDouble() : 0,
                            RfdNPerS = metrics.TryGetProperty("rfd_N_per_s", out var rfd) ? (float)rfd.GetDouble() : 0,
                            TakeoffFrame = metrics.TryGetProperty("takeoff_frame", out var to) && to.ValueKind != JsonValueKind.Null ? to.GetInt32() : null
                        };
                    }
                    
                    // Parse time series (optional, can be large)
                    if (root.TryGetProperty("grf_timeseries", out var ts))
                    {
                        grfResult.GrfTimeseries = new GRFTimeSeries();
                        
                        if (ts.TryGetProperty("time_s", out var timeArr))
                        {
                            var times = new float[timeArr.GetArrayLength()];
                            int i = 0;
                            foreach (var t in timeArr.EnumerateArray())
                            {
                                times[i++] = (float)t.GetDouble();
                            }
                            grfResult.GrfTimeseries.TimeS = times;
                        }
                        
                        if (ts.TryGetProperty("grf_vertical_N", out var grfArr))
                        {
                            var grfVals = new float[grfArr.GetArrayLength()];
                            int i = 0;
                            foreach (var g in grfArr.EnumerateArray())
                            {
                                grfVals[i++] = (float)g.GetDouble();
                            }
                            grfResult.GrfTimeseries.GrfVerticalN = grfVals;
                        }
                    }
                    
                    return grfResult;
                }
                
                return new GRFEstimationResult { Success = false, Error = "GRF estimation failed" };
            }
            finally
            {
                // Cleanup temp file
                if (File.Exists(outputPath))
                {
                    try { File.Delete(outputPath); } catch { }
                }
            }
        }

        /// <summary>
        /// Runs OpenSim model scaling using Pose2Sim.
        /// </summary>
        /// <param name="trcPath">Path to TRC file</param>
        /// <param name="outputDir">Output directory for scaled model</param>
        /// <param name="heightM">Subject height in meters</param>
        /// <param name="massKg">Subject mass in kg</param>
        /// <param name="poseModel">Pose model name (default: COCO_133)</param>
        /// <returns>Path to scaled .osim model file</returns>
        public async Task<(bool Success, string? ScaledModelPath, string? Error)> ScaleModelAsync(
            string trcPath, string outputDir, float heightM, float massKg, string poseModel = "COCO_133")
        {
            var args = $"scale --trc \"{trcPath}\" --output \"{outputDir}\" --height {heightM} --mass {massKg} --model {poseModel}";
            var result = await RunCommandAsync(args);
            
            if (!result.Success)
            {
                return (false, null, result.Error);
            }
            
            if (result.Output != null)
            {
                using var doc = JsonDocument.Parse(result.Output);
                var root = doc.RootElement;
                if (root.TryGetProperty("success", out var s) && s.GetBoolean())
                {
                    var modelPath = root.TryGetProperty("scaled_model", out var mp) ? mp.GetString() : null;
                    return (true, modelPath, null);
                }
                else
                {
                    var error = root.TryGetProperty("error", out var e) ? e.GetString() : "Unknown error";
                    return (false, null, error);
                }
            }
            
            return (false, null, "No output from scaling command");
        }

        /// <summary>
        /// Runs OpenSim Inverse Kinematics using Pose2Sim.
        /// </summary>
        /// <param name="trcPath">Path to TRC file</param>
        /// <param name="outputDir">Output directory (must contain scaled .osim model)</param>
        /// <param name="poseModel">Pose model name (default: COCO_133)</param>
        /// <returns>Path to .mot motion file</returns>
        public async Task<(bool Success, string? MotionFilePath, string? Error)> RunInverseKinematicsAsync(
            string trcPath, string outputDir, string poseModel = "COCO_133")
        {
            var args = $"ik --trc \"{trcPath}\" --output \"{outputDir}\" --model {poseModel}";
            var result = await RunCommandAsync(args);
            
            if (!result.Success)
            {
                return (false, null, result.Error);
            }
            
            if (result.Output != null)
            {
                using var doc = JsonDocument.Parse(result.Output);
                var root = doc.RootElement;
                if (root.TryGetProperty("success", out var s) && s.GetBoolean())
                {
                    var motPath = root.TryGetProperty("motion_file", out var mf) ? mf.GetString() : null;
                    return (true, motPath, null);
                }
                else
                {
                    var error = root.TryGetProperty("error", out var e) ? e.GetString() : "Unknown error";
                    return (false, null, error);
                }
            }
            
            return (false, null, "No output from IK command");
        }

        /// <summary>
        /// Runs the complete CMJ analysis workflow:
        /// 1. Scale model
        /// 2. Run IK
        /// 3. Run BodyKinematics (CoM calculation)
        /// 4. Estimate GRF
        /// </summary>
        /// <param name="trcPath">Path to TRC file</param>
        /// <param name="outputDir">Output directory for all results</param>
        /// <param name="heightM">Subject height in meters</param>
        /// <param name="massKg">Subject mass in kg</param>
        /// <param name="poseModel">Pose model name (default: COCO_133)</param>
        /// <returns>GRF estimation result</returns>
        public async Task<GRFEstimationResult> RunFullCMJAnalysisAsync(
            string trcPath, string outputDir, float heightM, float massKg, string poseModel = "COCO_133")
        {
            // Ensure output directory exists
            Directory.CreateDirectory(outputDir);
            
            // Step 1: Scale model
            var scaleResult = await ScaleModelAsync(trcPath, outputDir, heightM, massKg, poseModel);
            if (!scaleResult.Success)
            {
                return new GRFEstimationResult { Success = false, Error = $"Scaling failed: {scaleResult.Error}" };
            }
            
            // Step 2: Run IK
            var ikResult = await RunInverseKinematicsAsync(trcPath, outputDir, poseModel);
            if (!ikResult.Success)
            {
                return new GRFEstimationResult { Success = false, Error = $"IK failed: {ikResult.Error}" };
            }
            
            // Step 3: Run BodyKinematics
            var bodykinCsvPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(trcPath) + "_bodykin.csv");
            var bodykinResult = await RunBodyKinematicsAsync(ikResult.MotionFilePath!, scaleResult.ScaledModelPath!, bodykinCsvPath);
            if (!bodykinResult.Success)
            {
                return new GRFEstimationResult { Success = false, Error = $"BodyKinematics failed: {bodykinResult.Error}" };
            }
            
            // Step 4: Estimate GRF from CoM data
            var grfResult = await EstimateGRFAsync(bodykinResult.OutputCsvPath!, massKg);
            return grfResult;
        }

        private async Task<(bool Success, string? Output, string? Error)> RunCommandAsync(string arguments)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = $"\"{_wrapperScriptPath}\" {arguments}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = new Process { StartInfo = psi };
                process.Start();

                var output = await process.StandardOutput.ReadToEndAsync();
                var error = await process.StandardError.ReadToEndAsync();

                await process.WaitForExitAsync();

                if (!string.IsNullOrWhiteSpace(output))
                {
                    // Extract only JSON from output (OpenSim may print info messages)
                    var jsonOutput = ExtractJson(output);
                    if (!string.IsNullOrEmpty(jsonOutput))
                    {
                        return (true, jsonOutput, null);
                    }
                }
                
                if (process.ExitCode != 0 || !string.IsNullOrWhiteSpace(error))
                {
                    return (false, null, string.IsNullOrWhiteSpace(error) ? "No valid JSON output" : error);
                }
                
                return (false, null, "No output from command");
            }
            catch (Exception ex)
            {
                return (false, null, $"Failed to run Python: {ex.Message}");
            }
        }

        /// <summary>
        /// Extracts JSON object from output that may contain other text.
        /// </summary>
        private static string? ExtractJson(string output)
        {
            // Find the last line that starts with '{'
            var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            for (int i = lines.Length - 1; i >= 0; i--)
            {
                var line = lines[i].Trim();
                if (line.StartsWith("{") && line.EndsWith("}"))
                {
                    return line;
                }
            }
            return null;
        }

        /// <summary>
        /// Reads BodyKinematics CSV file and returns CoM, CoP, and Contact Spheres positions.
        /// </summary>
        public (List<System.Numerics.Vector3> CoM, 
                List<System.Numerics.Vector3> CoP,
                Dictionary<string, List<System.Numerics.Vector3>> ContactSpheres) LoadBodyKinematicsData(string csvPath)
        {
            var comPositions = new List<System.Numerics.Vector3>();
            var copPositions = new List<System.Numerics.Vector3>();
            var contactSpheresData = new Dictionary<string, List<System.Numerics.Vector3>>();
            
            try
            {
                var lines = File.ReadAllLines(csvPath);
                if (lines.Length < 2) return (comPositions, copPositions, contactSpheresData);

                // Parse header to find column indices
                char separator = lines[0].Contains('\t') ? '\t' : ',';
                var headers = lines[0].Split(separator)
                    .Select(h => h.Trim().TrimStart('#').Trim().ToLowerInvariant())
                    .ToList();

                int idxX = headers.IndexOf("com_x");
                int idxY = headers.IndexOf("com_y");
                int idxZ = headers.IndexOf("com_z");
                
                // CoP indices
                int copX = headers.IndexOf("cop_x");
                int copY = headers.IndexOf("cop_y");
                int copZ = headers.IndexOf("cop_z");
                bool hasCoP = copX != -1 && copY != -1 && copZ != -1;

                // Contact Spheres indices map: Name -> (idxX, idxY, idxZ)
                var sphereIndices = new Dictionary<string, (int x, int y, int z)>();
                foreach (var header in headers)
                {
                    if (header.StartsWith("cs_") && header.EndsWith("_x"))
                    {
                        string name = header.Substring(3, header.Length - 5); // cs_..._x
                        int sx = headers.IndexOf($"cs_{name}_x");
                        int sy = headers.IndexOf($"cs_{name}_y");
                        int sz = headers.IndexOf($"cs_{name}_z");
                        
                        if (sx != -1 && sy != -1 && sz != -1)
                        {
                            sphereIndices[name] = (sx, sy, sz);
                            contactSpheresData[name] = new List<System.Numerics.Vector3>();
                        }
                    }
                }

                if (idxX == -1 || idxY == -1 || idxZ == -1) return (comPositions, copPositions, contactSpheresData); 

                // Parse data
                for (int i = 1; i < lines.Length; i++)
                {
                    if (string.IsNullOrWhiteSpace(lines[i])) continue;
                    var values = lines[i].Split(separator);
                    
                    // Parse CoM
                    if (values.Length > Math.Max(idxX, Math.Max(idxY, idxZ)))
                    {
                        if (float.TryParse(values[idxX], out float x) &&
                            float.TryParse(values[idxY], out float y) &&
                            float.TryParse(values[idxZ], out float z))
                        {
                            comPositions.Add(new System.Numerics.Vector3(x, y, z));
                        }
                        else comPositions.Add(System.Numerics.Vector3.Zero);
                    }
                    else comPositions.Add(System.Numerics.Vector3.Zero);
                    
                    // Parse CoP
                    if (hasCoP)
                    {
                        if (values.Length > Math.Max(copX, Math.Max(copY, copZ)) &&
                            float.TryParse(values[copX], out float cx) &&
                            float.TryParse(values[copY], out float cy) &&
                            float.TryParse(values[copZ], out float cz))
                        {
                            copPositions.Add(new System.Numerics.Vector3(cx, cy, cz));
                        }
                        else copPositions.Add(System.Numerics.Vector3.Zero);
                    }
                    
                    // Parse Contact Spheres
                    foreach (var kvp in sphereIndices)
                    {
                        var name = kvp.Key;
                        var idxs = kvp.Value;
                        if (values.Length > Math.Max(idxs.x, Math.Max(idxs.y, idxs.z)) &&
                            float.TryParse(values[idxs.x], out float sx) &&
                            float.TryParse(values[idxs.y], out float sy) &&
                            float.TryParse(values[idxs.z], out float sz))
                        {
                            contactSpheresData[name].Add(new System.Numerics.Vector3(sx, sy, sz));
                        }
                        else
                        {
                            contactSpheresData[name].Add(System.Numerics.Vector3.Zero);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error reading body kinematics CSV: {ex.Message}");
            }
            return (comPositions, copPositions, contactSpheresData);
        }
    }
}
