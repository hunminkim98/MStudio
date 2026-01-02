using System;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using MStudio.Core.Interfaces;
using MStudio.Core.Messaging;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    /// <summary>
    /// Main application ViewModel - orchestrates the application state.
    /// 
    /// Clean Architecture Notes:
    /// - Uses IMessenger (WeakReferenceMessenger) for ViewModel-to-ViewModel communication
    /// - Does not directly call methods on child ViewModels for data refresh
    /// - Publishes MarkerDataChangedMessage after data modifications
    /// - Other ViewModels subscribe and react independently
    /// </summary>
    public partial class MainViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;
        private readonly IDialogService _dialogService;
        private readonly ITrialService _trialService;
        private readonly IExportService _exportService;
        private readonly IMovementAnalysisService _movementAnalysisService;
        private readonly ICMJAnalysisService _cmjAnalysisService;

        // Track selected marker locally (synced via messaging)
        [ObservableProperty]
        private int _selectedMarkerIndex = -1;

        [ObservableProperty]
        private string _statusText = "Ready";

        [ObservableProperty]
        private string _title = "MStudio - Motive Grade Workspace";

        // ViewportViewModel for 3D rendering (exposed for DataContext binding in XAML)
        public MStudioViewportViewModel ViewportViewModel { get; }
        
        // GraphViewModel for time-series data (exposed for DataContext binding in XAML)
        public GraphViewModel GraphViewModel { get; }

        // DataViewModel for file management (exposed for DataContext binding in XAML)
        public DataViewModel DataViewModel { get; }

        // Trial items for the Trials panel (left sidebar)
        public ObservableCollection<TrialItemViewModel> Trials { get; } = new();

        // Indicates if any trials are loaded
        public bool HasTrials => _trialService.HasTrials;

        public MainViewModel(
            ISessionService sessionService, 
            ITimelineService timelineService,
            IDialogService dialogService,
            ITrialService trialService,
            IExportService exportService,
            IMovementAnalysisService movementAnalysisService,
            ICMJAnalysisService cmjAnalysisService,
            MStudioViewportViewModel viewportViewModel, 
            GraphViewModel graphViewModel,
            DataViewModel dataViewModel)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;
            _dialogService = dialogService;
            _trialService = trialService;
            _exportService = exportService;
            _movementAnalysisService = movementAnalysisService;
            _cmjAnalysisService = cmjAnalysisService;
            ViewportViewModel = viewportViewModel;
            GraphViewModel = graphViewModel;
            DataViewModel = dataViewModel;

            // Re-broadcast property changes from services if needed
            _timelineService.PropertyChanged += (s, e) => 
            {
                if (e.PropertyName == nameof(ITimelineService.CurrentFrame))
                {
                    OnPropertyChanged(nameof(CurrentFrame));
                    OnPropertyChanged(nameof(DisplayFrame));
                }
                if (e.PropertyName == nameof(ITimelineService.TotalFrames))
                    OnPropertyChanged(nameof(TotalFrames));
                if (e.PropertyName == nameof(ITimelineService.IsPlaying))
                    OnPropertyChanged(nameof(IsPlaying));
                if (e.PropertyName == nameof(ITimelineService.CurrentTime))
                    OnPropertyChanged(nameof(CurrentTimeText));
            };

            _sessionService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ISessionService.CurrentMotion))
                {
                    OnPropertyChanged(nameof(CurrentMotionName));
                }
            };

            // Subscribe to trial collection changes
            _trialService.TrialsCollectionChanged += OnTrialsCollectionChanged;
            _trialService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ITrialService.HasTrials))
                {
                    OnPropertyChanged(nameof(HasTrials));
                }
            };

            // Subscribe to marker selection changes from child ViewModels
            WeakReferenceMessenger.Default.Register<MarkerSelectionChangedMessage>(this, (r, m) =>
            {
                // Update our local tracking (for FillGaps/SmoothData to use)
                // Use SetProperty to avoid triggering PropertyChanged unnecessarily
                if (m.Source != this && m.SelectedMarkerIndex != SelectedMarkerIndex)
                {
                    SetProperty(ref _selectedMarkerIndex, m.SelectedMarkerIndex, nameof(SelectedMarkerIndex));
                }
            });
        }

        public bool IsPlaying => _timelineService.IsPlaying;

        public int CurrentFrame
        {
            get => _timelineService.CurrentFrame;
            set => _timelineService.CurrentFrame = value;
        }

        /// <summary>
        /// 1-indexed frame number for UI display (Frame 0 displays as Frame 1).
        /// </summary>
        public int DisplayFrame => _timelineService.CurrentFrame + 1;

        public int TotalFrames => _timelineService.TotalFrames;

        public double PlaybackSpeed
        {
            get => _timelineService.PlaybackSpeed;
            set
            {
                _timelineService.PlaybackSpeed = Math.Clamp(value, 0.0, 10.0);
                OnPropertyChanged();
            }
        }

        public bool IsLooping
        {
            get => _timelineService.IsLooping;
            set
            {
                _timelineService.IsLooping = value;
                OnPropertyChanged();
            }
        }

        public bool IsReverse
        {
            get => _timelineService.IsReverse;
            set
            {
                _timelineService.IsReverse = value;
                OnPropertyChanged();
            }
        }

        public string CurrentTimeText
        {
            get
            {
                var t = _timelineService.CurrentTime;
                return $"{t.Minutes:D1}:{t.Seconds:D2}:{t.Milliseconds:D3}";
            }
        }

        public string CurrentMotionName => _sessionService.CurrentMotion != null 
            ? Path.GetFileName(_sessionService.CurrentMotion.Metadata.FilePath) 
            : "No Motion Loaded";

        [RelayCommand]
        private void Play() => _timelineService.Play();

        [RelayCommand]
        private void Pause() => _timelineService.Pause();

        [RelayCommand]
        private void TogglePlay() => _timelineService.TogglePlay();

        [RelayCommand]
        private void ResetSpeed() => PlaybackSpeed = 1.0;

        [RelayCommand]
        private void StepForward() => _timelineService.StepForward();

        [RelayCommand]
        private void StepBackward() => _timelineService.StepBackward();

        /// <summary>
        /// Opens the analysis selection dialog and runs the selected analysis.
        /// </summary>
        [RelayCommand]
        private async Task OpenAnalysis()
        {
            // Show the analysis selection dialog
            var selectedType = _dialogService.ShowAnalysisSelectionDialog();
            
            if (selectedType == null)
                return;

            var analysisType = selectedType.Value;
            
            // Special handling for CMJ analysis
            if (analysisType == AnalysisType.CounterMovementJump)
            {
                await RunCMJAnalysis();
                return;
            }

            // Special handling for BodyKinematics analysis
            if (analysisType == AnalysisType.BodyKinematics)
            {
                await RunBodyKinematicsAnalysis();
                return;
            }

            var typeInfo = _movementAnalysisService.GetAvailableAnalysisTypes()
                .FirstOrDefault(t => t.Type == analysisType);

            if (typeInfo == null)
                return;

            // Run mock analysis for other types
            var result = await _movementAnalysisService.RunAnalysisAsync(analysisType, _sessionService.CurrentMotion);

            // Show result in a message box (mock implementation)
            _dialogService.ShowInfo(result.Summary, $"{typeInfo.DisplayName} Analysis");
            StatusText = $"{typeInfo.DisplayName} analysis completed";
        }

        /// <summary>
        /// Runs BodyKinematics (OpenSim) analysis.
        /// </summary>
        private async Task RunBodyKinematicsAnalysis()
        {
            // Check if motion data is loaded
            if (_sessionService.CurrentMotion == null)
            {
                _dialogService.ShowError("No motion data loaded. Please load a TRC file first.", "BodyKinematics Analysis");
                return;
            }

            // Show input dialog
            var inputDialog = new Views.BodyKinematicsInputDialog
            {
                Owner = System.Windows.Application.Current.MainWindow
            };

            if (inputDialog.ShowDialog() != true || !inputDialog.Confirmed)
                return;

            StatusText = "Running OpenSim BodyKinematics analysis...";

            try
            {
                var trcPath = _sessionService.CurrentMotion.Metadata.FilePath;
                var outputDir = Path.Combine(Path.GetDirectoryName(trcPath) ?? Path.GetTempPath(), "opensim_output");
                Directory.CreateDirectory(outputDir);

                // Create Pose2Sim service from config
                var pose2sim = MStudio.Services.Implementations.Pose2SimWrapperService.CreateFromConfig();

                // Check availability
                StatusText = "Checking OpenSim/Pose2Sim availability...";
                var (available, version, error) = await pose2sim.CheckAvailabilityAsync();
                if (!available)
                {
                    _dialogService.ShowError($"Pose2Sim not available: {error}", "BodyKinematics Analysis");
                    StatusText = "BodyKinematics analysis failed - Pose2Sim not available";
                    return;
                }

                // Step 1: Scale model
                StatusText = "Step 1/3: Scaling OpenSim model...";
                var scaleResult = await pose2sim.ScaleModelAsync(trcPath, outputDir, inputDialog.HeightM, inputDialog.BodyMassKg, "HALPE_26");
                if (!scaleResult.Success)
                {
                    _dialogService.ShowError($"Model scaling failed: {scaleResult.Error}", "BodyKinematics Analysis");
                    StatusText = "BodyKinematics analysis failed - Scaling error";
                    return;
                }

                // Step 2: Run IK
                StatusText = "Step 2/3: Running Inverse Kinematics...";
                var ikResult = await pose2sim.RunInverseKinematicsAsync(trcPath, outputDir, "HALPE_26");
                if (!ikResult.Success)
                {
                    _dialogService.ShowError($"Inverse Kinematics failed: {ikResult.Error}", "BodyKinematics Analysis");
                    StatusText = "BodyKinematics analysis failed - IK error";
                    return;
                }

                // Step 3: Run BodyKinematics
                StatusText = "Step 3/3: Calculating Body Kinematics (CoM)...";
                var bodykinCsvPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(trcPath) + "_bodykin.csv");
                var bodykinResult = await pose2sim.RunBodyKinematicsAsync(ikResult.MotionFilePath!, scaleResult.ScaledModelPath!, bodykinCsvPath);
                if (!bodykinResult.Success)
                {
                    _dialogService.ShowError($"BodyKinematics failed: {bodykinResult.Error}", "BodyKinematics Analysis");
                    StatusText = "BodyKinematics analysis failed";
                    return;
                }

                // Success!
                _dialogService.ShowInfo(
                    $"BodyKinematics analysis completed successfully!\n\n" +
                    $"Output files saved to:\n{outputDir}\n\n" +
                    $"- Scaled Model: {Path.GetFileName(scaleResult.ScaledModelPath)}\n" +
                    $"- Motion File: {Path.GetFileName(ikResult.MotionFilePath)}\n" +
                    $"- BodyKinematics: {Path.GetFileName(bodykinCsvPath)}\n\n" +
                    $"Frames: {bodykinResult.FrameRate:F1} fps",
                    "BodyKinematics Analysis Complete"
                );

                StatusText = $"BodyKinematics analysis completed - Output saved to {outputDir}";
            }
            catch (Exception ex)
            {
                _dialogService.ShowError($"BodyKinematics analysis failed: {ex.Message}", "BodyKinematics Analysis Error");
                StatusText = "BodyKinematics analysis failed";
            }
        }

        /// <summary>
        /// Runs CMJ analysis with input dialog and result window.
        /// </summary>
        private async Task RunCMJAnalysis()
        {
            // Check if motion data is loaded
            if (_sessionService.CurrentMotion == null)
            {
                _dialogService.ShowError("No motion data loaded. Please load a file first.", "CMJ Analysis");
                return;
            }

            // Check if required markers exist
            if (!_cmjAnalysisService.HasRequiredMarkers(_sessionService.CurrentMotion))
            {
                _dialogService.ShowWarning("Missing required markers for CMJ analysis (Hip, RHip, LHip, RKnee, LKnee, RAnkle, LAnkle).", "CMJ Analysis");
            }

            // Show input dialog
            var inputDialog = new Views.CMJInputDialog
            {
                Owner = System.Windows.Application.Current.MainWindow
            };

            if (inputDialog.ShowDialog() != true || !inputDialog.Confirmed)
                return;

            string? bodyKinCsvPath = null;
            
            // If Estimate GRF is selected, show file selection dialog
            if (inputDialog.EstimateGRF)
            {
                var grfFilesDialog = new Views.GRFFilesInputDialog
                {
                    Owner = System.Windows.Application.Current.MainWindow
                };

                if (grfFilesDialog.ShowDialog() != true || !grfFilesDialog.Confirmed)
                    return;

                bodyKinCsvPath = grfFilesDialog.BodyKinematicsCsvPath;
            }

            StatusText = inputDialog.EstimateGRF 
                ? "Running CMJ analysis with GRF estimation..." 
                : "Running CMJ analysis...";

            try
            {
                // First run standard CMJ analysis
                var result = await _cmjAnalysisService.AnalyzeAsync(
                    _sessionService.CurrentMotion,
                    inputDialog.SelectedGender,
                    inputDialog.BodyMassKg);

                // If Estimate GRF is selected and we have a valid file, run GRF estimation
                if (inputDialog.EstimateGRF && !string.IsNullOrEmpty(bodyKinCsvPath))
                {
                    StatusText = "Loading BodyKinematics data...";
                    
                    try
                    {
                        var pose2sim = MStudio.Services.Implementations.Pose2SimWrapperService.CreateFromConfig();

                        // Step 1: Load CoM, CoP, and Contact Spheres data from CSV
                        var (bodyKinCoM, bodyKinCoP, bodyKinContactSpheres) = pose2sim.LoadBodyKinematicsData(bodyKinCsvPath);
                        
                        // Step 2: Phase Detection with OpenSim CoM (논문 기반 알고리즘)
                        int takeoffFrame = result.TakeoffFrame;
                        int landingFrame = result.LandingFrame;
                        int lowestCoMFrame = result.LowestCoMFrame;
                        int peakFlightFrame = result.PeakFlightFrame;
                        int movementStartFrame = result.MovementStartFrame;
                        int brakingStartFrame = result.BrakingStartFrame;
                        int landingDepthFrame = result.LandingDepthFrame;
                        float frameRate = _sessionService.CurrentMotion.Metadata.FrameRate;
                        IReadOnlyList<CMJPhaseInfo> phases = result.Phases;

                        if (bodyKinCoM.Count > 10)
                        {
                            StatusText = "Detecting CMJ phases with OpenSim CoM...";
                            
                            // Use new CMJPhaseDetectionService with OpenSim CoM
                            var phaseService = new MStudio.Services.Implementations.CMJPhaseDetectionService();
                            var phaseResult = phaseService.DetectPhases(
                                bodyKinCoM, 
                                _sessionService.CurrentMotion, 
                                frameRate);

                            if (phaseResult.IsSuccess)
                            {
                                takeoffFrame = phaseResult.TakeoffFrame;
                                landingFrame = phaseResult.LandingFrame;
                                lowestCoMFrame = phaseResult.LowestCoMFrame;
                                peakFlightFrame = phaseResult.PeakHeightFrame;
                                movementStartFrame = phaseResult.MovementStartFrame;
                                brakingStartFrame = phaseResult.BrakingStartFrame;
                                landingDepthFrame = phaseResult.LandingDepthFrame;
                                phases = phaseResult.Phases;
                                
                                StatusText = $"Phase detection: LowestCoM={lowestCoMFrame}, Take-off={takeoffFrame}, Landing={landingFrame}";
                            }
                        }

                        // Step 3: GRF Estimation (사용 업데이트된 프레임 정보)
                        StatusText = "Estimating GRF from BodyKinematics data...";
                        var grfResult = await pose2sim.EstimateGRFAsync(
                            bodyKinCsvPath, 
                            inputDialog.BodyMassKg,
                            takeoffFrame > 0 ? takeoffFrame : null,
                            landingFrame > 0 ? landingFrame : null,
                            lowestCoMFrame > 0 ? lowestCoMFrame : null);

                        if (grfResult.Success && grfResult.Metrics != null)
                        {
                            // Clean Flight Phase GRF to 0 based on Kinematic Events
                            var grfSeries = grfResult.GrfTimeseries?.GrfVerticalN?.ToList() ?? new System.Collections.Generic.List<float>();
                            int flightStart = takeoffFrame + 1;
                            int flightEnd = landingFrame - 1;

                            if (flightStart > 0 && flightEnd > flightStart)
                            {
                                for (int i = flightStart; i <= flightEnd; i++)
                                {
                                    if (i >= 0 && i < grfSeries.Count)
                                    {
                                        grfSeries[i] = 0f;
                                    }
                                }
                            }

                            // Create new result with GRF data and updated phases
                            result = new CMJAnalysisResult
                            {
                                Type = result.Type,
                                IsSuccess = result.IsSuccess,
                                AnalyzedAt = result.AnalyzedAt,
                                Summary = result.Summary + " (with GRF & OpenSim CoM)",
                                
                                SubjectGender = result.SubjectGender,
                                SubjectMassKg = result.SubjectMassKg,
                                
                                LowestCoMFrame = lowestCoMFrame,
                                TakeoffFrame = takeoffFrame,
                                LandingFrame = landingFrame,
                                PeakFlightFrame = peakFlightFrame,
                                
                                // 논문 기반 추가 프레임
                                MovementStartFrame = movementStartFrame,
                                BrakingStartFrame = brakingStartFrame,
                                LandingDepthFrame = landingDepthFrame,
                                FrameRate = frameRate,
                                
                                HipKneeRatio = result.HipKneeRatio,
                                Dominance = result.Dominance,
                                HipMomentEstimate = result.HipMomentEstimate,
                                KneeMomentEstimate = result.KneeMomentEstimate,
                                
                                LeftKneeValgus = result.LeftKneeValgus,
                                RightKneeValgus = result.RightKneeValgus,
                                
                                Phases = phases,
                                TimeSeries = result.TimeSeries,
                                CoMPositions = bodyKinCoM.Count > 0 ? bodyKinCoM : result.CoMPositions,
                                CoPPositions = bodyKinCoP.Count > 0 ? bodyKinCoP : result.CoPPositions,
                                ContactSpheresTrajectories = bodyKinContactSpheres.Count > 0 
                                    ? bodyKinContactSpheres.ToDictionary(k => k.Key, v => (IReadOnlyList<System.Numerics.Vector3>)v.Value) 
                                    : result.ContactSpheresTrajectories,
                                
                                // Jump Performance - 새로운 프레임 정보로 재계산
                                JumpHeightMeters = CalculateJumpHeight(bodyKinCoM, peakFlightFrame, movementStartFrame),
                                FlightTimeSeconds = (landingFrame - takeoffFrame) / frameRate,
                                ContactTimeSeconds = (takeoffFrame - movementStartFrame) / frameRate,

                                // GRF data
                                HasGRFData = true,
                                PeakVerticalGRF_N = grfResult.Metrics.PeakVerticalGrfN,
                                NetVerticalImpulse_Ns = grfResult.Metrics.NetVerticalImpulseNs,
                                RFD_NPerS = grfResult.Metrics.RfdNPerS,
                                GRFTimeSeries = grfSeries,
                                GRFTimeValues = grfResult.GrfTimeseries?.TimeS?.ToList() ?? new System.Collections.Generic.List<float>()
                            };
                        }
                        else
                        {
                            _dialogService.ShowWarning($"GRF estimation failed: {grfResult.Error}\n\nContinuing with standard CMJ analysis.", "GRF Estimation");
                        }
                    }
                    catch (Exception grfEx)
                    {
                        _dialogService.ShowWarning($"GRF estimation error: {grfEx.Message}\n\nContinuing with standard CMJ analysis.", "GRF Estimation");
                    }
                }

                // Show result window
                var resultWindow = new Views.CMJResultWindow(result)
                {
                    Owner = System.Windows.Application.Current.MainWindow
                };
                
                // Send message to viewport for visualization
                WeakReferenceMessenger.Default.Send(new CMJAnalysisCompletedMessage 
                { 
                    Result = result, 
                    ShowVisualization = true 
                });
                
                // Clear visualization when window closes
                resultWindow.Closed += (sender, args) =>
                {
                    WeakReferenceMessenger.Default.Send(new CMJAnalysisCompletedMessage 
                    { 
                        Result = result, 
                        ShowVisualization = false 
                    });
                };
                
                resultWindow.Show();

                var grfStatus = result.HasGRFData ? " with GRF data" : "";
                StatusText = $"CMJ Analysis completed - {result.Dominance}{grfStatus}";
            }
            catch (Exception ex)
            {
                _dialogService.ShowError($"CMJ Analysis failed: {ex.Message}", "CMJ Analysis Error");
                StatusText = "CMJ Analysis failed";
            }
        }


        /// <summary>
        /// Fills gaps in the selected marker's data using interpolation.
        /// Uses message-based communication to notify ViewModels of data change.
        /// </summary>
        [RelayCommand]
        private void FillGaps()
        {
            var motion = _sessionService.CurrentMotion;
            
            if (motion != null && SelectedMarkerIndex >= 0)
            {
                motion.Markers.FillGaps(SelectedMarkerIndex, 100); // Default 100 frames
                
                // Notify all interested ViewModels via messaging (Clean Architecture)
                WeakReferenceMessenger.Default.Send(new MarkerDataChangedMessage
                {
                    MarkerIndex = SelectedMarkerIndex,
                    ChangeDescription = "Gap filling completed"
                });
                
                StatusText = $"Filled gaps for marker {SelectedMarkerIndex}";
            }
        }

        /// <summary>
        /// Smooths the selected marker's data using a moving average filter.
        /// Uses message-based communication to notify ViewModels of data change.
        /// </summary>
        [RelayCommand]
        private void SmoothData()
        {
            var motion = _sessionService.CurrentMotion;

            if (motion != null && SelectedMarkerIndex >= 0)
            {
                motion.Markers.SmoothData(SelectedMarkerIndex, 5); // Default 5 frame window

                // Notify all interested ViewModels via messaging (Clean Architecture)
                WeakReferenceMessenger.Default.Send(new MarkerDataChangedMessage
                {
                    MarkerIndex = SelectedMarkerIndex,
                    ChangeDescription = "Data smoothing completed"
                });
                
                StatusText = $"Smoothed data for marker {SelectedMarkerIndex}";
            }
        }

        /// <summary>
        /// Deletes the currently selected marker after confirmation.
        /// </summary>
        [RelayCommand]
        private void DeleteSelectedMarker()
        {
            if (SelectedMarkerIndex < 0) return;
            
            // Prioritize TrialService if available and has selected trials
            // This aligns with ViewportViewModel's visualization prioritization
            if (_trialService != null && _trialService.HasSelectedTrials && _trialService.SelectedTrials.Count > 0)
            {
                var activeTrial = _trialService.SelectedTrials[0];
                var motion = activeTrial.MotionData;
                
                string markerName = "Unknown";
                if (SelectedMarkerIndex >= 0 && SelectedMarkerIndex < motion.Metadata.MarkerNames.Count)
                {
                    markerName = motion.Metadata.MarkerNames[SelectedMarkerIndex];
                }

                bool confirmed = _dialogService.ShowConfirmation(
                    $"Are you sure you want to delete marker '{markerName}' from trial '{activeTrial.Name}'?", 
                    "Delete Marker");

                if (confirmed)
                {
                    int indexToDelete = SelectedMarkerIndex;
                    
                    // Reset selection BEFORE deleting
                    SelectedMarkerIndex = -1;
                    
                    _trialService.DeleteMarker(activeTrial.Id, indexToDelete);
                    StatusText = $"Deleted marker: {markerName}";
                }
            }
            // Fallback to SessionService
            else if (_sessionService.CurrentMotion != null)
            {
                var motion = _sessionService.CurrentMotion;
                
                string markerName = "Unknown";
                if (SelectedMarkerIndex >= 0 && SelectedMarkerIndex < motion.Metadata.MarkerNames.Count)
                {
                    markerName = motion.Metadata.MarkerNames[SelectedMarkerIndex];
                }

                bool confirmed = _dialogService.ShowConfirmation(
                    $"Are you sure you want to delete marker '{markerName}'?", 
                    "Delete Marker");

                if (confirmed)
                {
                    int indexToDelete = SelectedMarkerIndex;
                    
                    // Reset selection BEFORE deleting
                    SelectedMarkerIndex = -1;
                    
                    _sessionService.DeleteMarker(indexToDelete);
                    StatusText = $"Deleted marker: {markerName}";
                    OnPropertyChanged(nameof(CurrentMotionName));
                }
            }
        }

        /// <summary>
        /// Opens a file using the abstracted dialog service.
        /// This maintains Clean Architecture by not depending on platform-specific UI elements.
        /// Now adds the file as a trial to the TrialService.
        /// </summary>
        [RelayCommand]
        private async Task OpenFile()
        {
            const string filter = "Motion Files (*.trc; *.c3d; *.json)|*.trc;*.c3d;*.json|All files (*.*)|*.*";
            
            var filePath = _dialogService.ShowOpenFileDialog(filter, "Open Motion File");
            
            if (filePath != null)
            {
                StatusText = $"Loading {Path.GetFileName(filePath)}...";
                try
                {
                    // Add as a trial (primary method for multi-trial support)
                    var trial = await _trialService.AddTrialAsync(filePath);
                    
                    // Also load into session for backward compatibility
                    await _sessionService.LoadMotionAsync(filePath);
                    
                    StatusText = $"Added trial: {trial.Name}";
                }
                catch (Exception ex)
                {
                    _dialogService.ShowError($"Failed to load file: {ex.Message}", "Load Error");
                    StatusText = $"Error: {ex.Message}";
                }
            }
        }

        /// <summary>
        /// Handles trial collection changes and syncs with TrialItemViewModel collection.
        /// </summary>
        private void OnTrialsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
        {
            switch (e.Action)
            {
                case NotifyCollectionChangedAction.Add:
                    if (e.NewItems != null)
                    {
                        foreach (Trial trial in e.NewItems)
                        {
                            Trials.Add(new TrialItemViewModel(trial, _trialService));
                        }
                    }
                    break;
                    
                case NotifyCollectionChangedAction.Remove:
                    if (e.OldItems != null)
                    {
                        foreach (Trial trial in e.OldItems)
                        {
                            var vm = Trials.FirstOrDefault(t => t.Id == trial.Id);
                            if (vm != null)
                            {
                                vm.Dispose();
                                Trials.Remove(vm);
                            }
                        }
                    }
                    break;
                    
                case NotifyCollectionChangedAction.Reset:
                    // Make a copy to avoid modifying collection while iterating
                    var viewModelsToDispose = Trials.ToList();
                    Trials.Clear();
                    
                    // Dispose after clearing to avoid event handler issues
                    foreach (var vm in viewModelsToDispose)
                    {
                        try
                        {
                            vm.Dispose();
                        }
                        catch
                        {
                            // Ignore dispose errors during clear
                        }
                    }
                    break;
            }
            
            OnPropertyChanged(nameof(HasTrials));
        }

        /// <summary>
        /// Removes a trial by ID.
        /// </summary>
        [RelayCommand]
        private void RemoveTrial(string? trialId)
        {
            if (trialId != null)
            {
                _trialService.RemoveTrial(trialId);
            }
        }

        /// <summary>
        /// Clears all trials.
        /// </summary>
        [RelayCommand]
        private void ClearAllTrials()
        {
            _trialService.ClearTrials();
        }

        /// <summary>
        /// Selects all trials.
        /// </summary>
        [RelayCommand]
        private void SelectAllTrials()
        {
            _trialService.SelectAllTrials();
        }

        /// <summary>
        /// Deselects all trials.
        /// </summary>
        [RelayCommand]
        private void DeselectAllTrials()
        {
            _trialService.DeselectAllTrials();
        }

        /// <summary>
        /// Saves a trial to its original file path (overwrite).
        /// </summary>
        [RelayCommand]
        private async Task SaveTrial(TrialItemViewModel? trialVm)
        {
            if (trialVm == null) return;
            await _exportService.SaveAsync(trialVm.Trial);
        }

        /// <summary>
        /// Opens a save dialog and saves the trial to the selected path.
        /// </summary>
        [RelayCommand]
        private async Task SaveTrialAs(TrialItemViewModel? trialVm)
        {
            if (trialVm == null) return;
            await _exportService.SaveAsAsync(trialVm.Trial);
        }

        /// <summary>
        /// Calculates jump height from CoM positions.
        /// Jump height = Peak CoM height - Initial standing CoM height
        /// </summary>
        private static float CalculateJumpHeight(
            IReadOnlyList<System.Numerics.Vector3> comPositions, 
            int peakFrame, 
            int standingFrame)
        {
            if (comPositions == null || comPositions.Count == 0)
                return 0f;

            // Use first 10 frames average as standing height
            float standingHeight = 0f;
            int validCount = 0;
            int avgFrames = Math.Min(10, comPositions.Count);
            
            for (int i = 0; i < avgFrames; i++)
            {
                if (comPositions[i].Y > 0.01f)
                {
                    standingHeight += comPositions[i].Y;
                    validCount++;
                }
            }
            standingHeight = validCount > 0 ? standingHeight / validCount : 0;

            // Peak height
            float peakHeight = 0f;
            if (peakFrame >= 0 && peakFrame < comPositions.Count)
            {
                peakHeight = comPositions[peakFrame].Y;
            }
            else
            {
                // Find max if peakFrame is invalid
                foreach (var pos in comPositions)
                {
                    if (pos.Y > peakHeight) peakHeight = pos.Y;
                }
            }

            return Math.Max(0, peakHeight - standingHeight);
        }
    }
}

