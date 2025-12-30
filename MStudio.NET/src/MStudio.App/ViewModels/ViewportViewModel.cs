using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Media.Media3D;
using System.Numerics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using HelixToolkit;
using HelixToolkit.Geometry;
using HelixToolkit.Maths;
using HelixToolkit.SharpDX;
using HelixToolkit.Wpf.SharpDX;
using MStudio.Core.Interfaces;
using MStudio.Core.Messaging;
using MStudio.Core.Models;
using MStudio.Services.Implementations;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    public partial class MStudioViewportViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;
        private readonly IVisualizationSettingsService _visualizationSettings;
        private readonly ITrialService? _trialService;
        private readonly IFootLevelingService? _footLevelingService;
        private readonly IAnalysisService _analysisService;

        // 3D Scene Elements
        public ObservableCollection<Element3D> SceneElements { get; } = new();

        private LineGeometryModel3D? _gridModel;
        private GroupModel3D? _originModel;

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(UpAxisText))]
        private bool _isZUp = false;

        public string UpAxisText => IsZUp ? "Z" : "Y";

        [ObservableProperty]
        private bool _isShowGrid = true;

        [ObservableProperty]
        private bool _isShowOriginAxis = true;

        [ObservableProperty]
        private bool _isShowRuler = false;

        [ObservableProperty]
        private bool _isShowTrajectory = true;

        partial void OnIsShowTrajectoryChanged(bool value)
        {
            if (_trajectoryModel != null) _trajectoryModel.IsRendering = value;
        }

        // Foot Contact Visualization
        [ObservableProperty]
        private bool _isShowFootContact = false;

        partial void OnIsShowFootContactChanged(bool value)
        {
            if (_leftFootContactModel != null) _leftFootContactModel.IsRendering = value && _isLeftFootContacting;
            if (_rightFootContactModel != null) _rightFootContactModel.IsRendering = value && _isRightFootContacting;
            
            if (value)
            {
                CalculateFootMinLevels();
                UpdateFootContactVisuals();
            }
            if (value)
            {
                CalculateFootMinLevels();
                UpdateFootContactVisuals();
            }
        }

        // Analysis Mode
        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(AnalysisResultText))]
        private bool _isAnalysisMode = false;

        partial void OnIsAnalysisModeChanged(bool value)
        {
            if (value)
            {
                // Enter analysis mode: clear standard selection
                SelectedMarkerIndex = -1;
                AnalysisSelectedMarkers.Clear();
                AnalysisResultText = "Select markers to analyze...";
                UpdateAnalysisOverlay();
            }
            else
            {
                // Exit analysis mode: clear analysis selection
                AnalysisSelectedMarkers.Clear();
                AnalysisResultText = "";
                UpdateAnalysisOverlay();
            }
        }

        public ObservableCollection<int> AnalysisSelectedMarkers { get; } = new();

        [ObservableProperty]
        private string _analysisResultText = "";

        [ObservableProperty]
        private AnalysisAxis _selectedAnalysisAxis = AnalysisAxis.Y; // Default to Y-axis (Vertical)

        partial void OnSelectedAnalysisAxisChanged(AnalysisAxis value)
        {
            UpdateAnalysisOverlay();
        }

        [RelayCommand]
        private void ToggleAnalysisMode()
        {
            IsAnalysisMode = !IsAnalysisMode;
        }

        // Analysis Visualization Models
        private InstancingMeshGeometryModel3D? _analysisMarkersModel; // Orange (Target/Default)
        private InstancingMeshGeometryModel3D? _analysisRefMarkersModel; // Cyan (Reference)
        private BillboardTextModel3D? _analysisLabelModel;
        private LineGeometryModel3D? _analysisLinesModel;
        private MeshGeometryModel3D? _analysisArcMeshModel; // Filled Fan for Arc
        
        // Analysis Settings
        private readonly Color4 _analysisMarkerColor = new Color4(1.0f, 0.4f, 0.0f, 1.0f); // Orange
        private readonly Color4 _analysisRefMarkerColor = new Color4(0.0f, 1.0f, 1.0f, 1.0f); // Cyan
        private readonly Color4 _analysisLineColor = new Color4(1.0f, 1.0f, 1.0f, 0.8f); // White lines

        private MeshGeometryModel3D? _leftFootContactModel;
        private MeshGeometryModel3D? _rightFootContactModel;
        
        // Ground levels (Lowest Average Y)
        private float _minLeftFootY = float.MaxValue;
        private float _minRightFootY = float.MaxValue;
        private bool _isLeftFootContacting = false;
        private bool _isRightFootContacting = false;

        [ObservableProperty]
        private Vector3D _modelUpDirection = new Vector3D(0, 1, 0);

        partial void OnIsShowGridChanged(bool value)
        {
            if (_gridModel != null) _gridModel.IsRendering = value;
        }

        partial void OnIsShowOriginAxisChanged(bool value)
        {
            if (_originModel != null) _originModel.IsRendering = value;
        }

        partial void OnIsShowRulerChanged(bool value)
        {
            if (_rulerLabelsModel != null) _rulerLabelsModel.IsRendering = value;
            if (_rulerMarkersModelX != null) _rulerMarkersModelX.IsRendering = value;
            if (_rulerMarkersModelOther != null) _rulerMarkersModelOther.IsRendering = value;
        }

        partial void OnIsZUpChanged(bool value)
        {
            // Re-create Grid and Axis to match orientation
            if (_gridModel != null) SceneElements.Remove(_gridModel);
            if (_originModel != null) SceneElements.Remove(_originModel);
            
            CreateGrid();
            CreateAxis();
            CreateRulerLabels();

            // Adjust Camera and Control orientation
            var up = value ? new Vector3D(0, 0, 1) : new Vector3D(0, 1, 0);
            if (Camera != null)
            {
                Camera.UpDirection = up;
            }
            ModelUpDirection = up;
        }

        [RelayCommand]
        private void ResetView()
        {
            if (Camera != null)
            {
                Camera.Position = new Point3D(2, 2, 2);
                Camera.LookDirection = new Vector3D(-2, -2, -2);
                Camera.UpDirection = IsZUp ? new Vector3D(0, 0, 1) : new Vector3D(0, 1, 0);
            }
        }

        // ========== Foot Leveling ==========
        
        /// <summary>
        /// Foot leveling이 현재 적용되어 있는지 여부 (UI 바인딩용)
        /// </summary>
        [ObservableProperty]
        private bool _isFootLevelingApplied;

        /// <summary>
        /// Foot leveling 토글 (Apply ↔ Undo)
        /// </summary>
        [RelayCommand]
        private void ToggleFootLeveling()
        {
            if (_footLevelingService == null) return;

            // TrialService 사용 시 선택된 Trial의 MotionData 사용
            MotionData? motionData = null;
            if (_trialService != null && _trialService.SelectedTrials.Count > 0)
            {
                // 첫 번째 선택된 Trial에 적용 (모든 Trial에 동일 오프셋 적용)
                motionData = _trialService.SelectedTrials[0].MotionData;
            }
            else
            {
                motionData = _sessionService.CurrentMotion;
            }

            if (motionData == null) return;

            if (_footLevelingService.IsApplied)
            {
                _footLevelingService.UndoFootLeveling(motionData);
                
                // 다른 선택된 Trial들에도 Undo 적용
                if (_trialService != null)
                {
                    for (int i = 1; i < _trialService.SelectedTrials.Count; i++)
                    {
                        var trialMotion = _trialService.SelectedTrials[i].MotionData;
                        if (trialMotion != null)
                        {
                            // 동일 오프셋으로 복원
                            for (int frame = 0; frame < trialMotion.Markers.FrameCount; frame++)
                            {
                                for (int marker = 0; marker < trialMotion.Markers.MarkerCount; marker++)
                                {
                                    var pos = trialMotion.Markers.GetPosition(marker, frame);
                                    if (!float.IsNaN(pos.X))
                                    {
                                        trialMotion.Markers.SetPosition(marker, frame, pos.X, pos.Y + _footLevelingService.AppliedOffset, pos.Z);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                // 발 마커가 없으면 안내 메시지 표시
                bool success = _footLevelingService.ApplyFootLeveling(motionData);
                if (!success)
                {
                    System.Windows.MessageBox.Show(
                        "현재 모델에는 발 마커(BigToe, SmallToe, Heel)가 없습니다.\n\n" +
                        "이 기능을 사용하려면 발 마커가 포함된 키포인트셋을 사용해 주세요.\n" +
                        "예: HALPE_26, BODY_25, COCO_133 등",
                        "Foot Leveling 불가",
                        System.Windows.MessageBoxButton.OK,
                        System.Windows.MessageBoxImage.Information);
                    return;
                }

                // 다른 선택된 Trial들에도 동일 오프셋 적용
                if (_trialService != null)
                {
                    float offset = _footLevelingService.AppliedOffset;
                    for (int i = 1; i < _trialService.SelectedTrials.Count; i++)
                    {
                        var trialMotion = _trialService.SelectedTrials[i].MotionData;
                        if (trialMotion != null)
                        {
                            for (int frame = 0; frame < trialMotion.Markers.FrameCount; frame++)
                            {
                                for (int marker = 0; marker < trialMotion.Markers.MarkerCount; marker++)
                                {
                                    var pos = trialMotion.Markers.GetPosition(marker, frame);
                                    if (!float.IsNaN(pos.X))
                                    {
                                        trialMotion.Markers.SetPosition(marker, frame, pos.X, pos.Y - offset, pos.Z);
                                    }
                                }
                            }
                        }
                    }
                }

                // 조정된 오프셋 알림
                System.Windows.MessageBox.Show(
                    $"Y축 오프셋 {_footLevelingService.AppliedOffset:F4} m 만큼 조정되었습니다.",
                    "Set to Zero 완료",
                    System.Windows.MessageBoxButton.OK,
                    System.Windows.MessageBoxImage.Information);
            }

            IsFootLevelingApplied = _footLevelingService.IsApplied;
            
            // Recalculate Min Levels as Y values changed
            CalculateFootMinLevels();
            
            UpdateMarkersAndVisuals();
        }

        // Marker Rendering
        private InstancingMeshGeometryModel3D? _markerModel;
        private InstancingMeshGeometryModel3D? _selectedMarkerModel;
        private HelixToolkit.SharpDX.MeshGeometry3D? _markerSphereGeometry;
        private HelixToolkit.SharpDX.MeshGeometry3D? _selectedMarkerSphereGeometry;
        
        // Selected marker highlight settings
        private const float SelectedMarkerSizeMultiplier = 1.5f;
        private static readonly Color4 SelectedMarkerHighlightColor = new Color4(1.0f, 0.85f, 0.2f, 1.0f); // Bright gold

        // Ruler Labels and Markers
        private BillboardTextModel3D? _rulerLabelsModel;
        private InstancingMeshGeometryModel3D? _rulerMarkersModelX; // Red Axis
        private InstancingMeshGeometryModel3D? _rulerMarkersModelOther; // Yellow or Blue Axis

        // Marker Names (billboard text)
        private BillboardTextModel3D? _markerNamesModel;

        [ObservableProperty]
        private bool _isShowMarkerNames = false;

        partial void OnIsShowMarkerNamesChanged(bool value)
        {
            if (_markerNamesModel != null) _markerNamesModel.IsRendering = value;
        }

        // Trajectory & Bones
        private LineGeometryModel3D? _trajectoryModel;
        private LineGeometryModel3D? _boneModel;

        [ObservableProperty]
        private int _selectedMarkerIndex = -1;

        // Skeleton model selection
        public ObservableCollection<string> AvailableSkeletons { get; } = new();

        [ObservableProperty]
        private string _selectedSkeleton = "HALPE_26";

        partial void OnSelectedSkeletonChanged(string value)
        {
            // Regenerate bones when skeleton model changes
            if (_sessionService.CurrentMotion != null)
            {
                _boneLinks.Clear();
                AutoGenerateBones(_sessionService.CurrentMotion);
                UpdateBones();
            }
        }

        // Simple bone linkage (pair of marker indices)
        private readonly List<(int start, int end)> _boneLinks = new();

        private IEffectsManager? _effectsManager;
        public IEffectsManager? EffectsManager
        {
            get => _effectsManager;
            set => SetProperty(ref _effectsManager, value);
        }

        private HelixToolkit.Wpf.SharpDX.Camera? _camera;
        public HelixToolkit.Wpf.SharpDX.Camera? Camera
        {
            get => _camera;
            set => SetProperty(ref _camera, value);
        }

        [ObservableProperty]
        private string _markerCountText = "";

        [ObservableProperty]
        private string _frameInfoText = "";

        // List of all marker names in current motion
        public ObservableCollection<string> MarkerNames { get; } = new();

        // ========== Visualization Settings (Customizable) ==========
        
        /// <summary>
        /// Marker radius in meters. Range: 0.005 ~ 0.05
        /// </summary>
        [ObservableProperty]
        private float _markerSize = 0.012f;

        partial void OnMarkerSizeChanged(float value)
        {
            // Rebuild marker geometry with new size
            RebuildMarkerGeometry();
            // Sync to service
            _visualizationSettings.MarkerSize = value;
        }

        /// <summary>
        /// Marker opacity. Range: 0.1 ~ 1.0
        /// </summary>
        [ObservableProperty]
        private float _markerOpacity = 1.0f;

        partial void OnMarkerOpacityChanged(float value)
        {
            // Update marker color with new opacity
            MarkerColor = new Color4(MarkerColor.Red, MarkerColor.Green, MarkerColor.Blue, value);
            // Sync to service
            _visualizationSettings.MarkerOpacity = value;
        }

        /// <summary>
        /// Marker color (RGBA)
        /// </summary>
        [ObservableProperty]
        private Color4 _markerColor = new Color4(0.2f, 0.8f, 0.3f, 1.0f); // Green

        partial void OnMarkerColorChanged(Color4 value)
        {
            UpdateMarkerMaterial();
            OnPropertyChanged(nameof(MarkerColorDisplay));
            // Sync to service (without opacity, opacity is separate)
            _visualizationSettings.MarkerColor = (value.Red, value.Green, value.Blue, value.Alpha);
        }

        /// <summary>
        /// Marker color as WPF Color for UI display binding
        /// </summary>
        public System.Windows.Media.Color MarkerColorDisplay => 
            System.Windows.Media.Color.FromArgb(
                (byte)(MarkerColor.Alpha * 255),
                (byte)(MarkerColor.Red * 255),
                (byte)(MarkerColor.Green * 255),
                (byte)(MarkerColor.Blue * 255));

        /// <summary>
        /// Selected marker color for trajectory display
        /// </summary>
        [ObservableProperty]
        private System.Windows.Media.Color _selectedMarkerColor = System.Windows.Media.Color.FromArgb(255, 255, 230, 100);

        partial void OnSelectedMarkerColorChanged(System.Windows.Media.Color value)
        {
            if (_trajectoryModel != null)
            {
                _trajectoryModel.Color = TrajectoryColor;
            }
            UpdateTrajectories();
        }

        /// <summary>
        /// Bone/skeleton line color
        /// </summary>
        [ObservableProperty]
        private System.Windows.Media.Color _boneColor = System.Windows.Media.Color.FromArgb(180, 200, 200, 200);

        partial void OnBoneColorChanged(System.Windows.Media.Color value)
        {
            if (_boneModel != null)
            {
                _boneModel.Color = value;
            }
            // Sync to service
            _visualizationSettings.BoneColor = (value.ScR, value.ScG, value.ScB, value.ScA);
        }

        /// <summary>
        /// Bone line thickness
        /// </summary>
        [ObservableProperty]
        private double _boneThickness = 1.5;

        partial void OnBoneThicknessChanged(double value)
        {
            if (_boneModel != null)
            {
                _boneModel.Thickness = value;
            }
            // Sync to service
            _visualizationSettings.BoneThickness = value;
        }

        /// <summary>
        /// Bone opacity. Range: 0.1 ~ 1.0
        /// </summary>
        [ObservableProperty]
        private float _boneOpacity = 0.8f;

        partial void OnBoneOpacityChanged(float value)
        {
            // Update bone color with new opacity
            BoneColor = System.Windows.Media.Color.FromScRgb(value, BoneColor.ScR, BoneColor.ScG, BoneColor.ScB);
            // Sync to service
            _visualizationSettings.BoneOpacity = value;
        }

        // Trajectory color (derived from selected marker color)
        private System.Windows.Media.Color TrajectoryColor => 
            System.Windows.Media.Color.FromArgb(128, SelectedMarkerColor.R, SelectedMarkerColor.G, SelectedMarkerColor.B);

        public MStudioViewportViewModel(
            ISessionService sessionService,
            ITimelineService timelineService,
            IVisualizationSettingsService visualizationSettings,
            IAnalysisService analysisService,
            ITrialService? trialService = null,
            IFootLevelingService? footLevelingService = null)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;
            _visualizationSettings = visualizationSettings;
            _analysisService = analysisService;
            _trialService = trialService;
            _footLevelingService = footLevelingService;

            // Sync initial values from service
            SyncFromVisualizationSettings();

            EffectsManager = new DefaultEffectsManager();

            var meshBuilder = new MeshBuilder(true, false);
            meshBuilder.AddSphere(new Vector3(0, 0, 0), MarkerSize, 12, 12);
            _markerSphereGeometry = meshBuilder.ToMeshGeometry3D();
            
            // Create larger geometry for selected marker
            var selectedMeshBuilder = new MeshBuilder(true, false);
            selectedMeshBuilder.AddSphere(new Vector3(0, 0, 0), MarkerSize * SelectedMarkerSizeMultiplier, 16, 16);
            _selectedMarkerSphereGeometry = selectedMeshBuilder.ToMeshGeometry3D();

            Camera = new HelixToolkit.Wpf.SharpDX.PerspectiveCamera
            {
                Position = new Point3D(2, 2, 2),
                LookDirection = new Vector3D(-2, -2, -2),
                UpDirection = new Vector3D(0, 1, 0),
                FarPlaneDistance = 1000,
                NearPlaneDistance = 0.01
            };

            // Initialize available skeleton models
            foreach (var skeleton in PredefinedSkeletons.All)
            {
                AvailableSkeletons.Add(skeleton.Name);
            }

            CreateGrid();
            CreateAxis();
            CreateRulerLabels();
            CreateTrajectoryAndBoneModels();
            CreateFootContactModels();
            CreateAnalysisModels();

            _sessionService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ISessionService.CurrentMotion))
                {
                    OnMotionLoaded();
                }
            };

            _timelineService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ITimelineService.CurrentFrame))
                {
                    UpdateMarkersAndVisuals();
                }
            };

            // Subscribe to trial selection changes for multi-trial rendering
            if (_trialService != null)
            {
                _trialService.SelectionChanged += (s, e) =>
                {
                    OnTrialSelectionChanged();
                };
            }

            // Subscribe to visualization settings changes (Clean Architecture: Service -> ViewModel)
            _visualizationSettings.PropertyChanged += OnVisualizationSettingsChanged;

            // Register for marker data change messages (from MainViewModel after FillGaps/Smooth)
            WeakReferenceMessenger.Default.Register<MarkerDataChangedMessage>(this, (r, m) =>
            {
                UpdateMarkersAndVisuals();
            });

            // Register for marker selection sync (from other ViewModels)
            WeakReferenceMessenger.Default.Register<MarkerSelectionChangedMessage>(this, (r, m) =>
            {
                // Avoid re-sending if we are the source
                if (m.Source != this)
                {
                    SelectedMarkerIndex = m.SelectedMarkerIndex;
                }
            });
        }

        /// <summary>
        /// Called when trial selection changes. Updates visualization for all selected trials.
        /// </summary>
        private void OnTrialSelectionChanged()
        {
            // Re-initialize bones for the first selected trial (if any)
            if (_trialService?.SelectedTrials.Count > 0)
            {
                var firstTrial = _trialService.SelectedTrials[0];
                _boneLinks.Clear();
                AutoGenerateBones(firstTrial.MotionData);
            }
            
            UpdateMarkersAndVisuals();
        }

        /// <summary>
        /// Handles property changes from the visualization settings service.
        /// Clean Architecture: ViewModel reacts to Service layer changes.
        /// </summary>
        private void OnVisualizationSettingsChanged(object? sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            switch (e.PropertyName)
            {
                case nameof(IVisualizationSettingsService.MarkerSize):
                    // Use property to trigger OnMarkerSizeChanged (rebuilds geometry)
                    MarkerSize = _visualizationSettings.MarkerSize;
                    break;
                case nameof(IVisualizationSettingsService.MarkerColor):
                    var mc = _visualizationSettings.MarkerColor;
                    MarkerColor = new Color4(mc.R, mc.G, mc.B, MarkerOpacity);
                    break;
                case nameof(IVisualizationSettingsService.MarkerOpacity):
                    MarkerOpacity = _visualizationSettings.MarkerOpacity;
                    break;
                case nameof(IVisualizationSettingsService.SelectedMarkerColor):
                    var smc = _visualizationSettings.SelectedMarkerColor;
                    SelectedMarkerColor = System.Windows.Media.Color.FromScRgb(smc.A, smc.R, smc.G, smc.B);
                    break;
                case nameof(IVisualizationSettingsService.BoneThickness):
                    BoneThickness = _visualizationSettings.BoneThickness;
                    break;
                case nameof(IVisualizationSettingsService.BoneColor):
                    var bc = _visualizationSettings.BoneColor;
                    BoneColor = System.Windows.Media.Color.FromScRgb(BoneOpacity, bc.R, bc.G, bc.B);
                    break;
                case nameof(IVisualizationSettingsService.BoneOpacity):
                    BoneOpacity = _visualizationSettings.BoneOpacity;
                    break;
                case nameof(IVisualizationSettingsService.ShowMarkerNames):
                    IsShowMarkerNames = _visualizationSettings.ShowMarkerNames;
                    break;
                case nameof(IVisualizationSettingsService.CurrentColorScheme):
                    OnPropertyChanged(nameof(SelectedColorScheme));
                    break;
            }
        }

        /// <summary>
        /// Syncs ViewModel properties from the visualization settings service.
        /// Called once during initialization.
        /// </summary>
        private void SyncFromVisualizationSettings()
        {
            var mc = _visualizationSettings.MarkerColor;
            MarkerSize = _visualizationSettings.MarkerSize;
            MarkerOpacity = _visualizationSettings.MarkerOpacity;
            MarkerColor = new Color4(mc.R, mc.G, mc.B, MarkerOpacity);
            
            var smc = _visualizationSettings.SelectedMarkerColor;
            SelectedMarkerColor = System.Windows.Media.Color.FromScRgb(smc.A, smc.R, smc.G, smc.B);
            
            BoneThickness = _visualizationSettings.BoneThickness;
            BoneOpacity = _visualizationSettings.BoneOpacity;
            var bc = _visualizationSettings.BoneColor;
            BoneColor = System.Windows.Media.Color.FromScRgb(BoneOpacity, bc.R, bc.G, bc.B);
            IsShowMarkerNames = _visualizationSettings.ShowMarkerNames;
        }

        /// <summary>
        /// Updates the visualization settings service when ViewModel properties change.
        /// Clean Architecture: ViewModel -> Service layer sync.
        /// </summary>
        private void SyncToVisualizationSettings()
        {
            _visualizationSettings.MarkerSize = MarkerSize;
            _visualizationSettings.MarkerColor = (MarkerColor.Red, MarkerColor.Green, MarkerColor.Blue, MarkerColor.Alpha);
            _visualizationSettings.MarkerOpacity = MarkerColor.Alpha;
            _visualizationSettings.SelectedMarkerColor = (SelectedMarkerColor.ScR, SelectedMarkerColor.ScG, SelectedMarkerColor.ScB, SelectedMarkerColor.ScA);
            _visualizationSettings.BoneThickness = BoneThickness;
            _visualizationSettings.BoneColor = (BoneColor.ScR, BoneColor.ScG, BoneColor.ScB, BoneColor.ScA);
            _visualizationSettings.BoneOpacity = BoneColor.ScA;
            _visualizationSettings.ShowMarkerNames = IsShowMarkerNames;
        }

        // ========== Color Presets (Service-backed) ==========

        /// <summary>
        /// Available color scheme names for UI binding
        /// </summary>
        public string[] AvailableColorSchemes => _visualizationSettings.GetAvailableColorSchemes();

        /// <summary>
        /// Currently selected color scheme
        /// </summary>
        public string SelectedColorScheme
        {
            get => _visualizationSettings.CurrentColorScheme;
            set
            {
                if (_visualizationSettings.CurrentColorScheme != value)
                {
                    _visualizationSettings.ApplyColorScheme(value);
                    OnPropertyChanged(nameof(SelectedColorScheme));
                }
            }
        }

        /// <summary>
        /// Reset all visualization settings to defaults
        /// </summary>
        [RelayCommand]
        private void ResetVisualizationSettings()
        {
            _visualizationSettings.ResetToDefaults();
        }

        private void CreateTrajectoryAndBoneModels()
        {
            _trajectoryModel = new LineGeometryModel3D
            {
                Color = TrajectoryColor,
                Thickness = 1.0,
                IsRendering = true
            };
            SceneElements.Add(_trajectoryModel);

            _boneModel = new LineGeometryModel3D
            {
                Color = BoneColor,
                Thickness = BoneThickness,
                IsRendering = true
            };
            SceneElements.Add(_boneModel);
        }

        /// <summary>
        /// Rebuilds the marker sphere geometry when size changes
        /// </summary>
        private void RebuildMarkerGeometry()
        {
            var meshBuilder = new MeshBuilder(true, false);
            meshBuilder.AddSphere(new Vector3(0, 0, 0), MarkerSize, 12, 12);
            _markerSphereGeometry = meshBuilder.ToMeshGeometry3D();
            
            // Rebuild selected marker geometry (larger size)
            var selectedMeshBuilder = new MeshBuilder(true, false);
            selectedMeshBuilder.AddSphere(new Vector3(0, 0, 0), MarkerSize * SelectedMarkerSizeMultiplier, 16, 16);
            _selectedMarkerSphereGeometry = selectedMeshBuilder.ToMeshGeometry3D();

            if (_markerModel != null)
            {
                _markerModel.Geometry = _markerSphereGeometry;
            }
            
            if (_selectedMarkerModel != null)
            {
                _selectedMarkerModel.Geometry = _selectedMarkerSphereGeometry;
            }
        }

        /// <summary>
        /// Updates the marker material when color changes
        /// </summary>
        private void UpdateMarkerMaterial()
        {
            if (_markerModel != null)
            {
                _markerModel.Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial 
                { 
                    DiffuseColor = MarkerColor 
                };
            }
        }


        
        /// <summary>
        /// Handles marker selection. In Analysis Mode, manages multi-selection.
        /// </summary>
        public void HandleMarkerSelection(int markerIndex)
        {
            if (markerIndex < 0) return;

            if (IsAnalysisMode)
            {
                // Analysis Mode: Multi-select up to 3 markers
                if (AnalysisSelectedMarkers.Contains(markerIndex))
                {
                    AnalysisSelectedMarkers.Remove(markerIndex);
                }
                else
                {
                    if (AnalysisSelectedMarkers.Count >= 4)
                    {
                        AnalysisSelectedMarkers.RemoveAt(0); // FIFO
                    }
                    AnalysisSelectedMarkers.Add(markerIndex);
                }
                UpdateAnalysisOverlay();
            }
            else
            {
                // Standard Mode: Single select
                SelectedMarkerIndex = markerIndex;
            }
        }

        partial void OnSelectedMarkerIndexChanged(int value)
        {
            // Only update standard visuals if NOT in analysis mode (or if clearing selection)
            if (!IsAnalysisMode || value == -1)
            {
                UpdateTrajectories();
                UpdateSelectedMarkerHighlight();
                
                // Publish selection change message
                WeakReferenceMessenger.Default.Send(new MarkerSelectionChangedMessage
                {
                    SelectedMarkerIndex = value,
                    Source = this
                });
            }
        }

        private void CreateGrid()
        {
            var builder = new LineBuilder();
            int gridSize = 15;
            float gridSpacing = 0.5f; // 0.5m per grid line

            if (IsZUp)
            {
                // Create grid lines (XY plane, Z=0)
                for (int i = -gridSize; i <= gridSize; i++)
                {
                    float pos = i * gridSpacing;
                    builder.AddLine(new Vector3(pos, -gridSize * gridSpacing, 0), new Vector3(pos, gridSize * gridSpacing, 0));
                    builder.AddLine(new Vector3(-gridSize * gridSpacing, pos, 0), new Vector3(gridSize * gridSpacing, pos, 0));
                }
            }
            else
            {
                // Create grid lines (XZ plane, Y=0)
                for (int i = -gridSize; i <= gridSize; i++)
                {
                    float pos = i * gridSpacing;
                    builder.AddLine(new Vector3(pos, 0, -gridSize * gridSpacing), new Vector3(pos, 0, gridSize * gridSpacing));
                    builder.AddLine(new Vector3(-gridSize * gridSpacing, 0, pos), new Vector3(gridSize * gridSpacing, 0, pos));
                }
            }

            var gridGeometry = builder.ToLineGeometry3D();
            _gridModel = new LineGeometryModel3D
            {
                Geometry = gridGeometry,
                Color = System.Windows.Media.Color.FromArgb(80, 100, 100, 100), // Subtler grid
                Thickness = 0.5,
                IsRendering = IsShowGrid
            };

            SceneElements.Add(_gridModel);
        }

        private void CreateAxis()
        {
            _originModel = new GroupModel3D();
            float axisLength = 0.3f; // Same as grid spacing
            float offset = 0.001f;   // Lift slightly above grid

            if (IsZUp)
            {
                // Z-up mode: Grid is on XY plane, so offset Z
                // X-Axis (Red)
                var xBuilder = new LineBuilder();
                xBuilder.AddLine(new Vector3(0, 0, offset), new Vector3(axisLength, 0, offset));
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = xBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.Red, Thickness = 2 });

                // Y-Axis (Yellow)
                var yBuilder = new LineBuilder();
                yBuilder.AddLine(new Vector3(0, 0, offset), new Vector3(0, axisLength, offset));
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = yBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.Yellow, Thickness = 2 });

                // Z-Axis (Blue) - Vertical
                var zBuilder = new LineBuilder();
                zBuilder.AddLine(new Vector3(0, 0, offset), new Vector3(0, 0, axisLength + offset));
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = zBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.Blue, Thickness = 2 });
            }
            else
            {
                // Y-up mode: Grid is on XZ plane, so offset Y
                // X-Axis (Red)
                var xBuilder = new LineBuilder();
                xBuilder.AddLine(new Vector3(0, offset, 0), new Vector3(axisLength, offset, 0));
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = xBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.Red, Thickness = 2 });

                // Y-Axis (Yellow) - Vertical
                var yBuilder = new LineBuilder();
                yBuilder.AddLine(new Vector3(0, offset, 0), new Vector3(0, axisLength + offset, 0));
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = yBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.Yellow, Thickness = 2 });

                // Z-Axis (Blue)
                var zBuilder = new LineBuilder();
                zBuilder.AddLine(new Vector3(0, offset, 0), new Vector3(0, offset, axisLength));
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = zBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.Blue, Thickness = 2 });
            }

            _originModel.IsRendering = IsShowOriginAxis;
            SceneElements.Add(_originModel);
        }

        private void CreateRulerLabels()
        {
            // 1. Clear existing models
            if (_rulerLabelsModel != null) SceneElements.Remove(_rulerLabelsModel);
            if (_rulerMarkersModelX != null) SceneElements.Remove(_rulerMarkersModelX);
            if (_rulerMarkersModelOther != null) SceneElements.Remove(_rulerMarkersModelOther);

            // 2. Initialize Models
            _rulerLabelsModel = new BillboardTextModel3D
            {
                IsRendering = IsShowRuler
            };

            // Ruler Marker Geometry (Small Sphere)
            var rulerMarkerBuilder = new MeshBuilder(true, false);
            rulerMarkerBuilder.AddSphere(Vector3.Zero, 0.02f, 8, 8); // Radius 0.02m
            var rulerMarkerGeometry = rulerMarkerBuilder.ToMeshGeometry3D();

            // Model for X-Axis (Red)
            _rulerMarkersModelX = new InstancingMeshGeometryModel3D
            {
                Geometry = rulerMarkerGeometry,
                IsRendering = IsShowRuler,
                Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = new Color4(1, 0, 0, 1) } // Red
            };

            // Model for Other Axis (Yellow or Blue)
            var otherColor = IsZUp ? new Color4(1, 1, 0, 1) : new Color4(0, 0, 1, 1); // Yellow or Blue
            _rulerMarkersModelOther = new InstancingMeshGeometryModel3D
            {
                Geometry = rulerMarkerGeometry,
                IsRendering = IsShowRuler,
                Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = otherColor }
            };

            // 3. Data Collection
            var textInfo = new BillboardText3D();
            var instancesX = new List<Matrix4x4>();
            var instancesOther = new List<Matrix4x4>();

            int gridSize = 15;
            float gridSpacing = 0.5f;

            float verticalOffset = 0.1f; // Lift text slightly off the grid
            float labelScale = 0.4f;      // Text size

            // Iterate 0.5m steps
            for (int i = 1; i <= gridSize; i++)
            {
                float dist = i * gridSpacing;
                string text = $"{dist:0.##}m";

                if (IsZUp)
                {
                    // === Z-UP System (Ground is XY) ===
                    
                    // X-Axis (Red)
                    var xPos = new Vector3(dist, 0, 0);
                    // Label
                    textInfo.TextInfo.Add(new TextInfo(text, xPos + new Vector3(0, 0, verticalOffset)) 
                    { 
                        Foreground = new Color4(1, 0, 0, 1), 
                        Scale = labelScale 
                    });
                    // Marker (X)
                    instancesX.Add(Matrix4x4.CreateTranslation(xPos));

                    // Y-Axis (Yellow)
                    var yPos = new Vector3(0, dist, 0);
                    // Label
                    textInfo.TextInfo.Add(new TextInfo(text, yPos + new Vector3(0, 0, verticalOffset)) 
                    { 
                        Foreground = new Color4(1, 1, 0, 1), 
                        Scale = labelScale 
                    });
                    // Marker (Other)
                    instancesOther.Add(Matrix4x4.CreateTranslation(yPos));
                }
                else
                {
                    // === Y-UP System (Ground is XZ) ===

                    // X-Axis (Red)
                    var xPos = new Vector3(dist, 0, 0);
                    // Label
                    textInfo.TextInfo.Add(new TextInfo(text, xPos + new Vector3(0, verticalOffset, 0)) 
                    { 
                        Foreground = new Color4(1, 0, 0, 1), 
                        Scale = labelScale 
                    });
                    // Marker (X)
                    instancesX.Add(Matrix4x4.CreateTranslation(xPos));

                    // Z-Axis (Blue)
                    var zPos = new Vector3(0, 0, dist);
                    // Label
                    textInfo.TextInfo.Add(new TextInfo(text, zPos + new Vector3(0, verticalOffset, 0)) 
                    { 
                        Foreground = new Color4(0, 0, 1, 1), 
                        Scale = labelScale 
                    });
                    // Marker (Other)
                    instancesOther.Add(Matrix4x4.CreateTranslation(zPos));
                }
            }
            
            // 4. Finalize Models
            _rulerLabelsModel.Geometry = textInfo;
            
            _rulerMarkersModelX.Instances = instancesX.ToArray();
            _rulerMarkersModelOther.Instances = instancesOther.ToArray();

            // Add to Scene
            SceneElements.Add(_rulerLabelsModel);
            SceneElements.Add(_rulerMarkersModelX);
            SceneElements.Add(_rulerMarkersModelOther);
        }

        private void OnMotionLoaded()
        {
            if (_markerModel != null) _markerModel.Instances = null;
            if (_selectedMarkerModel != null) _selectedMarkerModel.Instances = null;
            if (_trajectoryModel != null) _trajectoryModel.Geometry = null;
            if (_boneModel != null) _boneModel.Geometry = null;
            
            _boneLinks.Clear();
            MarkerNames.Clear();

            var motion = _sessionService.CurrentMotion;
            if (motion == null) return;

            // Populate marker names for UI list
            foreach (var name in motion.Metadata.MarkerNames)
            {
                MarkerNames.Add(name);
            }

            // Ensure marker model
            if (_markerModel == null)
            {
                _markerModel = new InstancingMeshGeometryModel3D
                {
                    Geometry = _markerSphereGeometry,
                    Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = MarkerColor }
                };
                SceneElements.Add(_markerModel);
            }
            
            // Ensure selected marker highlight model
            if (_selectedMarkerModel == null)
            {
                _selectedMarkerModel = new InstancingMeshGeometryModel3D
                {
                    Geometry = _selectedMarkerSphereGeometry,
                    Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = SelectedMarkerHighlightColor }
                };
                SceneElements.Add(_selectedMarkerModel);
            }

            // Ensure marker names billboard model
            if (_markerNamesModel == null)
            {
                _markerNamesModel = new BillboardTextModel3D
                {
                    IsRendering = IsShowMarkerNames
                };
                SceneElements.Add(_markerNamesModel);
            }

            AutoGenerateBones(motion);
            CalculateFootMinLevels();
            UpdateMarkersAndVisuals();

            MarkerCountText = $"Markers: {motion.Markers.MarkerCount}";
        }

        private void AutoGenerateBones(MotionData motion)
        {
            // Use predefined skeleton models based on marker names
            var names = motion.Metadata.MarkerNames;
            
            // Create a name-to-index lookup (case-insensitive)
            var nameToIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < names.Count; i++)
            {
                nameToIndex[names[i]] = i;
            }

            // Use the selected skeleton model
            var skeleton = PredefinedSkeletons.GetByName(SelectedSkeleton) ?? PredefinedSkeletons.Halpe26;
            
            // Build reverse lookup: joint name -> marker index
            var jointToMarkerIndex = new Dictionary<int, int>();
            foreach (var kvp in skeleton.JointMap)
            {
                int jointId = kvp.Key;
                string jointName = kvp.Value;
                
                if (nameToIndex.TryGetValue(jointName, out int markerIdx))
                {
                    jointToMarkerIndex[jointId] = markerIdx;
                }
            }

            // Create bone links from skeleton definition
            foreach (var bone in skeleton.Bones)
            {
                if (jointToMarkerIndex.TryGetValue(bone.Parent, out int parentIdx) &&
                    jointToMarkerIndex.TryGetValue(bone.Child, out int childIdx))
                {
                    _boneLinks.Add((parentIdx, childIdx));
                }
            }

            // If no bones were created from the skeleton template, fall back to proximity-based
            if (_boneLinks.Count == 0)
            {
                AutoGenerateBonesByProximity(motion);
            }
        }

        private void AutoGenerateBonesByProximity(MotionData motion)
        {
            // Fallback: proximity-based auto-skeleton for unknown marker sets
            var names = motion.Metadata.MarkerNames;
            for (int i = 0; i < names.Count; i++)
            {
                for (int j = i + 1; j < names.Count; j++)
                {
                    var p1 = motion.Markers.GetPosition(i, 0);
                    var p2 = motion.Markers.GetPosition(j, 0);

                    if (float.IsNaN(p1.X) || float.IsNaN(p2.X)) continue;

                    float dist = Vector3.Distance(p1, p2);
                    // Connect if markers are close (e.g. within 30cm) 
                    if (dist < 0.3f)
                    {
                        _boneLinks.Add((i, j));
                    }
                }
            }
        }

        public void UpdateMarkersAndVisuals()
        {
            UpdateMarkerPositions();
            UpdateSelectedMarkerHighlight();
            UpdateMarkerNames();
            UpdateFootContactVisuals();
            UpdateTrajectories();
            UpdateTrajectories();
            UpdateBones();
            UpdateAnalysisOverlay(); // Update analysis every frame
        }

        /// <summary>
        /// Updates marker positions for all selected trials or the current session motion.
        /// Supports multi-trial rendering with color coding and frame freeze for shorter trials.
        /// </summary>
        public void UpdateMarkerPositions()
        {
            // Use TrialService if available
            if (_trialService != null)
            {
                UpdateMarkerPositionsFromTrials();
                return;
            }
            
            // Fallback to session service (single motion)
            var motion = _sessionService.CurrentMotion;
            if (motion == null || _markerModel == null) return;

            int frame = _timelineService.CurrentFrame;
            if (frame < 0 || frame >= motion.Markers.FrameCount)
            {
                _markerModel.Instances = null;
                FrameInfoText = "";
                return;
            }

            FrameInfoText = $"Frame: {frame + 1} / {motion.Markers.FrameCount}";

            var instances = new Matrix4x4[motion.Markers.MarkerCount];
            for (int i = 0; i < motion.Markers.MarkerCount; i++)
            {
                var pos = motion.Markers.GetPosition(i, frame);
                if (float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    instances[i] = Matrix4x4.CreateScale(0);
                else
                    instances[i] = Matrix4x4.CreateTranslation(pos.X, pos.Y, pos.Z);
            }
            _markerModel.Instances = instances;
        }

        /// <summary>
        /// Updates marker positions from all selected trials.
        /// Combines markers from multiple trials with color coding.
        /// Implements frame freeze for shorter trials (displays last available frame).
        /// </summary>
        private void UpdateMarkerPositionsFromTrials()
        {
            if (_trialService == null || _markerModel == null) return;
            
            var selectedTrials = _trialService.SelectedTrials;
            if (selectedTrials.Count == 0)
            {
                _markerModel.Instances = null;
                FrameInfoText = "No trials selected";
                return;
            }

            int currentFrame = _timelineService.CurrentFrame;
            int maxFrame = _trialService.MaxSelectedFrameCount;
            
            FrameInfoText = $"Frame: {currentFrame + 1} / {maxFrame} ({selectedTrials.Count} trials)";

            // Count total markers across all selected trials
            int totalMarkers = selectedTrials.Sum(t => t.MarkerCount);
            var instances = new Matrix4x4[totalMarkers];
            
            int instanceIndex = 0;
            
            foreach (var trial in selectedTrials)
            {
                var motion = trial.MotionData;
                
                // Frame freeze: use last frame if current frame exceeds trial's frame count
                int trialFrame = Math.Min(currentFrame, motion.Markers.FrameCount - 1);
                if (trialFrame < 0) trialFrame = 0;
                
                for (int i = 0; i < motion.Markers.MarkerCount; i++)
                {
                    var pos = motion.Markers.GetPosition(i, trialFrame);
                    if (float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    {
                        instances[instanceIndex] = Matrix4x4.CreateScale(0);
                    }
                    else
                    {
                        instances[instanceIndex] = Matrix4x4.CreateTranslation(pos.X, pos.Y, pos.Z);
                    }
                    instanceIndex++;
                }
            }
            
            _markerModel.Instances = instances;
        }

        /// <summary>
        /// Gets the color for a trial based on its color index.
        /// </summary>
        private (float R, float G, float B) GetTrialColor(int colorIndex)
        {
            var palette = TrialService.TrialColorPalette;
            int idx = colorIndex % palette.Length;
            return palette[idx];
        }

        /// <summary>
        /// Updates the billboard text labels showing marker names in 3D space.
        /// </summary>
        public void UpdateMarkerNames()
        {
            // Check if we have any data to render
            MotionData? motion = null;
            
            if (_trialService?.HasSelectedTrials == true)
            {
                // Use first selected trial for marker names
                motion = _trialService.SelectedTrials[0].MotionData;
            }
            else if (_trialService == null)
            {
                // Fallback to session service
                motion = _sessionService.CurrentMotion;
            }
            
            if (motion == null || _markerNamesModel == null)
            {
                if (_markerNamesModel != null) _markerNamesModel.Geometry = null;
                return;
            }

            int frame = _timelineService.CurrentFrame;
            
            // Frame freeze for trials shorter than current frame
            if (frame >= motion.Markers.FrameCount)
            {
                frame = motion.Markers.FrameCount - 1;
            }
            if (frame < 0)
            {
                _markerNamesModel.Geometry = null;
                return;
            }

            // Create billboard text for each marker
            var textInfo = new BillboardText3D();
            var markerNames = motion.Metadata.MarkerNames;

            for (int i = 0; i < motion.Markers.MarkerCount; i++)
            {
                var pos = motion.Markers.GetPosition(i, frame);
                
                // Skip invalid markers
                if (float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    continue;

                string name = i < markerNames.Count ? markerNames[i] : $"M{i}";
                
                // Position text slightly above the marker
                var textPos = new Vector3(pos.X, pos.Y + 0.03f, pos.Z);
                
                textInfo.TextInfo.Add(new TextInfo(name, textPos)
                {
                    Foreground = Color4.White,
                    Scale = 0.4f
                });
            }

            _markerNamesModel.Geometry = textInfo;
        }

        /// <summary>
        /// Updates the selected marker highlight - renders a larger, brighter sphere at the selected marker position.
        /// </summary>
        public void UpdateSelectedMarkerHighlight()
        {
            if (_selectedMarkerModel == null)
            {
                return;
            }

            // No marker selected
            if (SelectedMarkerIndex < 0)
            {
                _selectedMarkerModel.Instances = null;
                return;
            }

            // Check if we have any data to render
            MotionData? motion = null;
            
            if (_trialService?.HasSelectedTrials == true)
            {
                // Use first selected trial for highlight
                motion = _trialService.SelectedTrials[0].MotionData;
            }
            else if (_trialService == null)
            {
                // Fallback to session service
                motion = _sessionService.CurrentMotion;
            }
            
            if (motion == null || SelectedMarkerIndex >= motion.Markers.MarkerCount)
            {
                _selectedMarkerModel.Instances = null;
                return;
            }

            int frame = _timelineService.CurrentFrame;
            
            // Frame freeze for trials shorter than current frame
            if (frame >= motion.Markers.FrameCount)
            {
                frame = motion.Markers.FrameCount - 1;
            }
            if (frame < 0)
            {
                _selectedMarkerModel.Instances = null;
                return;
            }

            var pos = motion.Markers.GetPosition(SelectedMarkerIndex, frame);
            
            // Skip invalid marker position
            if (float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0))
            {
                _selectedMarkerModel.Instances = null;
                return;
            }

            // Render single instance at the selected marker position
            _selectedMarkerModel.Instances = new[]
            {
                Matrix4x4.CreateTranslation(pos.X, pos.Y, pos.Z)
            };
        }

        public void UpdateTrajectories()
        {
            // Check if we have any data to render
            MotionData? motion = null;
            
            if (_trialService?.HasSelectedTrials == true)
            {
                // Use first selected trial for trajectory
                motion = _trialService.SelectedTrials[0].MotionData;
            }
            else if (_trialService == null)
            {
                // Fallback to session service
                motion = _sessionService.CurrentMotion;
            }
            
            if (motion == null || _trajectoryModel == null || SelectedMarkerIndex < 0)
            {
                if (_trajectoryModel != null) _trajectoryModel.Geometry = null;
                return;
            }

            int currentFrame = _timelineService.CurrentFrame;
            var builder = new LineBuilder();
            
            // Show trajectory for a window around current frame (e.g. +/- 60 frames)
            int start = Math.Max(0, currentFrame - 60);
            int end = Math.Min(motion.Markers.FrameCount - 1, currentFrame + 60);

            Vector3? lastPos = null;
            for (int f = start; f <= end; f++)
            {
                var pos = motion.Markers.GetPosition(SelectedMarkerIndex, f);
                if (float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                {
                    lastPos = null;
                    continue;
                }

                if (lastPos.HasValue)
                {
                    builder.AddLine(lastPos.Value, pos);
                }
                lastPos = pos;
            }

            _trajectoryModel.Geometry = builder.ToLineGeometry3D();
        }

        public void UpdateBones()
        {
            // Check if we have any data to render
            MotionData? motion = null;
            
            if (_trialService?.HasSelectedTrials == true)
            {
                // Use first selected trial for bones
                motion = _trialService.SelectedTrials[0].MotionData;
            }
            else if (_trialService == null)
            {
                // Fallback to session service
                motion = _sessionService.CurrentMotion;
            }
            
            if (motion == null || _boneModel == null || _boneLinks.Count == 0)
            {
                if (_boneModel != null) _boneModel.Geometry = null;
                return;
            }

            int frame = _timelineService.CurrentFrame;
            
            // Frame freeze for trials shorter than current frame
            if (frame >= motion.Markers.FrameCount)
            {
                frame = motion.Markers.FrameCount - 1;
            }
            if (frame < 0) frame = 0;
            
            var builder = new LineBuilder();

            foreach (var link in _boneLinks)
            {
                var p1 = motion.Markers.GetPosition(link.start, frame);
                var p2 = motion.Markers.GetPosition(link.end, frame);

                if (float.IsNaN(p1.X) || float.IsNaN(p2.X)) continue;
                if ((p1.X == 0 && p1.Y == 0 && p1.Z == 0) || (p2.X == 0 && p2.Y == 0 && p2.Z == 0)) continue;

                builder.AddLine(p1, p2);
            }

            _boneModel.Geometry = builder.ToLineGeometry3D();
        }

        /// <summary>
        /// Selects the nearest marker to the given 3D position.
        /// Called when user clicks in the 3D viewport.
        /// 
        /// Clean Architecture: This method contains the selection logic in the ViewModel,
        /// while the View only captures the click event and forwards the position.
        /// </summary>
        /// <param name="clickPosition">The 3D world position where the user clicked</param>
        /// <param name="maxDistance">Maximum distance to consider a marker as "clicked" (in meters)</param>
        public void SelectMarkerNearPosition(Vector3 clickPosition, float maxDistance = 0.1f)
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.MarkerCount == 0)
                return;

            int frame = _timelineService.CurrentFrame;
            if (frame < 0 || frame >= motion.Markers.FrameCount)
                return;

            int nearestMarkerIndex = -1;
            float nearestDistance = float.MaxValue;

            // Find the nearest marker to the click position
            for (int i = 0; i < motion.Markers.MarkerCount; i++)
            {
                var markerPos = motion.Markers.GetPosition(i, frame);
                
                // Skip invalid markers
                if (float.IsNaN(markerPos.X) || (markerPos.X == 0 && markerPos.Y == 0 && markerPos.Z == 0))
                    continue;

                float distance = Vector3.Distance(clickPosition, markerPos);
                
                if (distance < nearestDistance && distance <= maxDistance)
                {
                    nearestDistance = distance;
                    nearestMarkerIndex = i;
                }
            }

            // Select the nearest marker (if found)
            if (nearestMarkerIndex >= 0)
            {
                SelectedMarkerIndex = nearestMarkerIndex;
            }
        }

        /// <summary>
        /// Selects the nearest marker to a ray defined by an origin and direction.
        /// Used for object picking in 3D space.
        /// </summary>
        /// <param name="rayOrigin">Ray origin in world space</param>
        /// <param name="rayDirection">Normalized ray direction</param>
        /// <param name="maxRayDistance">Max distance from ray to consider a hit (meters)</param>
        /// <returns>True if a marker was selected</returns>
        public bool SelectMarkerByRay(Vector3 rayOrigin, Vector3 rayDirection, float maxRayDistance = 0.05f)
        {
            // TrialService 사용 시 선택된 Trial의 MotionData 사용
            MotionData? motion = null;
            if (_trialService != null && _trialService.SelectedTrials.Count > 0)
            {
                motion = _trialService.SelectedTrials[0].MotionData;
            }
            else
            {
                motion = _sessionService.CurrentMotion;
            }
            
            if (motion == null || motion.Markers.MarkerCount == 0)
                return false;

            int frame = _timelineService.CurrentFrame;
            if (frame < 0 || frame >= motion.Markers.FrameCount)
                return false;

            int bestMarkerIndex = -1;
            float minRayDistSq = maxRayDistance * maxRayDistance;
            float minProj = float.MaxValue;

            for (int i = 0; i < motion.Markers.MarkerCount; i++)
            {
                var markerPos = motion.Markers.GetPosition(i, frame);

                // Skip invalid markers
                if (float.IsNaN(markerPos.X) || (markerPos.X == 0 && markerPos.Y == 0 && markerPos.Z == 0))
                    continue;

                // Vector from ray origin to point
                Vector3 originToPoint = markerPos - rayOrigin;

                // Project point onto ray
                float proj = Vector3.Dot(originToPoint, rayDirection);

                // Skip if point is behind the camera
                if (proj < 0) continue;

                // Distance squared from point to ray
                // distSq = |originToPoint|^2 - proj^2
                float distSq = originToPoint.LengthSquared() - (proj * proj);

                // If within radius and closer than previous best (or closer to camera if multiple hits)
                if (distSq < minRayDistSq)
                {
                    // If multiple markers are close to the ray path, pick the one closest to camera
                    if (bestMarkerIndex == -1 || proj < minProj)
                    {
                        minRayDistSq = distSq;
                        minProj = proj;
                        bestMarkerIndex = i;
                    }
                }
            }

            if (bestMarkerIndex >= 0)
            {
                HandleMarkerSelection(bestMarkerIndex);
                return true;
            }

            return false;
        }
        private void CreateFootContactModels()
        {
            var builder = new MeshBuilder();
            
            // 1. Inner Core (Bright, concentrated contact point)
            // Taller (0.015m) and smaller radius (0.05m)
            builder.AddCylinder(Vector3.Zero, new Vector3(0, 0.015f, 0), 0.1f, 32);
            
            // 2. Outer Halo (Area of effect)
            // Thinner (0.005m) but wider radius (0.25m)
            builder.AddCylinder(Vector3.Zero, new Vector3(0, 0.005f, 0), 0.25f, 32);

            var geometry = builder.ToMeshGeometry3D();

            // High-Contrast Cyan Material (Reverted Color)
            var material = new HelixToolkit.Wpf.SharpDX.PhongMaterial
            {
                DiffuseColor = new Color4(0f, 1f, 1f, 0.9f),      // Bright Cyan
                EmissiveColor = new Color4(0f, 0.8f, 0.8f, 1.0f), // Glowing Effect
                SpecularColor = new Color4(1f, 1f, 1f, 1.0f),     // Shiny highlights
                SpecularShininess = 100f
            };

            _leftFootContactModel = new MeshGeometryModel3D
            {
                Geometry = geometry,
                Material = material,
                IsRendering = false, // Controlled by logic
                IsTransparent = true,
                CullMode = SharpDX.Direct3D11.CullMode.Back
            };

            _rightFootContactModel = new MeshGeometryModel3D
            {
                Geometry = geometry,
                Material = material,
                IsRendering = false,
                IsTransparent = true,
                CullMode = SharpDX.Direct3D11.CullMode.Back
            };

            SceneElements.Add(_leftFootContactModel);
            SceneElements.Add(_rightFootContactModel);
        }

        private void CalculateFootMinLevels()
        {
            _minLeftFootY = float.MaxValue;
            _minRightFootY = float.MaxValue;

            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers == null) return;

            // Define foot markers
            string[] leftFootNames = { "LBigToe", "LSmallToe", "LHeel" };
            string[] rightFootNames = { "RBigToe", "RSmallToe", "RHeel" };

            // Get indices
            var leftIndices = new List<int>();
            var rightIndices = new List<int>();

            for (int i = 0; i < motion.Metadata.MarkerNames.Count; i++)
            {
                string name = motion.Metadata.MarkerNames[i];
                if (leftFootNames.Any(n => string.Equals(n, name, StringComparison.OrdinalIgnoreCase))) leftIndices.Add(i);
                if (rightFootNames.Any(n => string.Equals(n, name, StringComparison.OrdinalIgnoreCase))) rightIndices.Add(i);
            }

            if (leftIndices.Count == 0 && rightIndices.Count == 0) return;

            // Helper to get avg Y
            float? GetAvgY(List<int> indices, int frame)
            {
                float sum = 0;
                int count = 0;
                foreach (var idx in indices)
                {
                    var pos = motion.Markers.GetPosition(idx, frame);
                    if (!float.IsNaN(pos.Y) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    {
                        sum += pos.Y;
                        count++;
                    }
                }
                return count > 0 ? sum / count : null;
            }

            // Scan frames
            for (int f = 0; f < motion.Markers.FrameCount; f++)
            {
                var lY = GetAvgY(leftIndices, f);
                if (lY.HasValue && lY.Value < _minLeftFootY) _minLeftFootY = lY.Value;

                var rY = GetAvgY(rightIndices, f);
                if (rY.HasValue && rY.Value < _minRightFootY) _minRightFootY = rY.Value;
            }
        }

        private void UpdateFootContactVisuals()
        {
            if (!IsShowFootContact || _sessionService.CurrentMotion == null)
            {
                if (_leftFootContactModel != null) _leftFootContactModel.IsRendering = false;
                if (_rightFootContactModel != null) _rightFootContactModel.IsRendering = false;
                return;
            }

            var motion = _sessionService.CurrentMotion;
            int frame = _timelineService.CurrentFrame;
            
            // Get current positions
             string[] leftFootNames = { "LBigToe", "LSmallToe", "LHeel" };
            string[] rightFootNames = { "RBigToe", "RSmallToe", "RHeel" };

            var markers = motion.Markers;
            var names = motion.Metadata.MarkerNames;

            // Helper to get center and contact status
            (bool isContact, Vector3 center) CheckFootContact(string[] targetNames, float minLevel)
            {
                float sumX = 0, sumZ = 0;
                int count = 0;
                bool contact = false;
                
                for(int i=0; i<names.Count; i++)
                {
                    if (targetNames.Any(n => string.Equals(n, names[i], StringComparison.OrdinalIgnoreCase)))
                    {
                        var pos = markers.GetPosition(i, frame);
                         if (!float.IsNaN(pos.Y) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                        {
                            sumX += pos.X;
                            sumZ += pos.Z;
                            count++;

                            // Check individual marker contact
                            if (minLevel != float.MaxValue && (pos.Y - minLevel) <= 0.0002f) // 2mm Threshold
                            {
                                contact = true;
                            }
                        }
                    }
                }

                if (count == 0) return (false, Vector3.Zero);
                return (contact, new Vector3(sumX / count, 0, sumZ / count)); // Y is unused for center
            }

            // Check Left
            var leftResult = CheckFootContact(leftFootNames, _minLeftFootY);
            if (leftResult.isContact)
            {
                _isLeftFootContacting = true;
                if (_leftFootContactModel != null)
                {
                    _leftFootContactModel.IsRendering = true;
                    // Lift slightly to avoid Z-fighting with floor
                    var translation = Matrix4x4.CreateTranslation(leftResult.center.X, 0.01f, leftResult.center.Z);
                    _leftFootContactModel.Transform = new MatrixTransform3D(translation.ToMatrix3D());
                }
            }
            else
            {
                _isLeftFootContacting = false;
                 if (_leftFootContactModel != null) _leftFootContactModel.IsRendering = false;
            }

            // Check Right
            var rightResult = CheckFootContact(rightFootNames, _minRightFootY);
            if (rightResult.isContact)
            {
                _isRightFootContacting = true;
                if (_rightFootContactModel != null)
                {
                    _rightFootContactModel.IsRendering = true;
                    // Lift slightly to avoid Z-fighting with floor
                    var translation = Matrix4x4.CreateTranslation(rightResult.center.X, 0.01f, rightResult.center.Z);
                    _rightFootContactModel.Transform = new MatrixTransform3D(translation.ToMatrix3D());
                }
            }
            else
            {
                _isRightFootContacting = false;
                if (_rightFootContactModel != null) _rightFootContactModel.IsRendering = false;
            }
        }
        
        private void CreateAnalysisModels()
        {
            // Analysis Markers (Orange Highlights)
            var sphereBuilder = new MeshBuilder(true, false);
            sphereBuilder.AddSphere(Vector3.Zero, MarkerSize * 1.5f, 16, 16);
            
            _analysisMarkersModel = new InstancingMeshGeometryModel3D
            {
                Geometry = sphereBuilder.ToMeshGeometry3D(),
                Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = _analysisMarkerColor },
                IsRendering = false
            };
            SceneElements.Add(_analysisMarkersModel);

            _analysisRefMarkersModel = new InstancingMeshGeometryModel3D
            {
                Geometry = sphereBuilder.ToMeshGeometry3D(),
                Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = _analysisRefMarkerColor },
                IsRendering = false
            };
            SceneElements.Add(_analysisRefMarkersModel);

            // Analysis Lines/Arcs
            _analysisLinesModel = new LineGeometryModel3D
            {
                Color = System.Windows.Media.Color.FromScRgb(0.3f, 1.0f, 1.0f, 0.0f), // Yellow with 0.3 Alpha
                Thickness = 2.0,
                IsRendering = false
            };
            SceneElements.Add(_analysisLinesModel);

            // Analysis Arc Mesh (Filled)
            _analysisArcMeshModel = new MeshGeometryModel3D
            {
                Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial 
                { 
                    DiffuseColor = new Color4(1.0f, 1.0f, 0.0f, 0.3f) // Yellow with 0.3 Alpha (same as lines) 
                },
                IsRendering = false,
                IsTransparent = true,
                CullMode = SharpDX.Direct3D11.CullMode.Back
            };
            SceneElements.Add(_analysisArcMeshModel);

            // Analysis Labels
            _analysisLabelModel = new BillboardTextModel3D
            {
                IsRendering = false
            };
            SceneElements.Add(_analysisLabelModel);
        }

        private void UpdateAnalysisOverlay()
        {
            if (!IsAnalysisMode)
            {
                if (_analysisMarkersModel != null) _analysisMarkersModel.IsRendering = false;
                if (_analysisRefMarkersModel != null) _analysisRefMarkersModel.IsRendering = false;
                if (_analysisLinesModel != null) _analysisLinesModel.IsRendering = false;
                if (_analysisArcMeshModel != null) _analysisArcMeshModel.IsRendering = false;
                if (_analysisLabelModel != null) _analysisLabelModel.IsRendering = false;
                return;
            }

            var motion = _sessionService.CurrentMotion;
            if (motion == null) return;

            int frame = _timelineService.CurrentFrame;
            float fps = motion.Metadata.FrameRate;
            if (fps <= 0) fps = 30.0f; // Fallback

            // 1. Update Selected Markers Highlight
            if (_analysisMarkersModel != null && _analysisRefMarkersModel != null)
            {
                var instancesTrg = new List<Matrix4x4>();
                var instancesRef = new List<Matrix4x4>();

                int totalSelected = AnalysisSelectedMarkers.Count;
                
                for (int i = 0; i < totalSelected; i++)
                {
                    int idx = AnalysisSelectedMarkers[i];
                    var pos = motion.Markers.GetPosition(idx, frame);
                    if (!float.IsNaN(pos.X))
                    {
                        var mat = Matrix4x4.CreateTranslation(pos.X, pos.Y, pos.Z);

                        // If 4 markers selected (Twist Mode), split 0,1 (Ref) and 2,3 (Target)
                        if (totalSelected == 4)
                        {
                            if (i < 2) instancesRef.Add(mat);
                            else instancesTrg.Add(mat);
                        }
                        else
                        {
                            // Otherwise all Orange (Target color)
                            instancesTrg.Add(mat);
                        }
                    }
                }
                
                _analysisMarkersModel.Instances = instancesTrg;
                _analysisMarkersModel.IsRendering = instancesTrg.Count > 0;

                _analysisRefMarkersModel.Instances = instancesRef;
                _analysisRefMarkersModel.IsRendering = instancesRef.Count > 0;
            }

            // 2. Perform Analysis & Update Visuals
            var lineBuilder = new LineBuilder();
            var meshBuilder = new MeshBuilder(false, false); // For Arc Fill
            var textInfo = new BillboardText3D();

            int count = AnalysisSelectedMarkers.Count;
            string resultText = "";

            if (count == 1)
            {
                // Velocity & Acceleration
                int idx = AnalysisSelectedMarkers[0];
                var posPrev = frame > 0 ? motion.Markers.GetPosition(idx, frame - 1) : new Vector3(float.NaN);
                var posCurr = motion.Markers.GetPosition(idx, frame);
                var posNext = frame < motion.Markers.FrameCount - 1 ? motion.Markers.GetPosition(idx, frame + 1) : new Vector3(float.NaN);

                if (!float.IsNaN(posCurr.X))
                {
                    // Velocity
                    var vel = _analysisService.CalculateVelocity(posPrev, posCurr, posNext, fps);
                    
                    // Acceleration (need adjacent velocities)
                    Vector3? acc = null;
                    if (frame > 1 && frame < motion.Markers.FrameCount - 2)
                    {
                        var posPrev2 = motion.Markers.GetPosition(idx, frame - 2);
                        var posNext2 = motion.Markers.GetPosition(idx, frame + 2);
                        
                        var velPrev = _analysisService.CalculateVelocity(posPrev2, posPrev, posCurr, fps);
                        var velNext = _analysisService.CalculateVelocity(posCurr, posNext, posNext2, fps);
                        
                        if (velPrev.HasValue && velNext.HasValue)
                        {
                            acc = _analysisService.CalculateAcceleration(velPrev.Value, velNext.Value, fps);
                        }
                    }

                    resultText = $"Marker: {motion.Metadata.MarkerNames[idx]}\n";
                    if (vel.HasValue) resultText += $"Vel: {vel.Value.Length():F3} m/s\n";
                    if (acc.HasValue) resultText += $"Acc: {acc.Value.Length():F3} m/s²";

                    // Visualize Velocity Vector? (Optional: draw arrow)
                    if (vel.HasValue)
                    {
                        lineBuilder.AddLine(posCurr, posCurr + Vector3.Normalize(vel.Value) * 0.2f); // 20cm arrow base
                    }
                }
            }
            else if (count == 2)
            {
                // Absolute Angle
                int idx1 = AnalysisSelectedMarkers[0];
                int idx2 = AnalysisSelectedMarkers[1];
                var p1 = motion.Markers.GetPosition(idx1, frame);
                var p2 = motion.Markers.GetPosition(idx2, frame);

                if (!float.IsNaN(p1.X) && !float.IsNaN(p2.X))
                {
                    lineBuilder.AddLine(p1, p2);
                    
                    float? angle = _analysisService.CalculateAbsoluteAngle(p1, p2, SelectedAnalysisAxis);
                    
                    if (angle.HasValue)
                    {
                        resultText = $"Absolute Angle ({SelectedAnalysisAxis}-axis)\n{angle.Value:F1}°";
                        
                        // Draw Axis Ref
                        Vector3 axisDir = SelectedAnalysisAxis switch
                        {
                            AnalysisAxis.X => Vector3.UnitX,
                            AnalysisAxis.Y => Vector3.UnitY,
                            AnalysisAxis.Z => Vector3.UnitZ, // Z-up handled by logic
                            _ => Vector3.UnitY
                        };
                        // Visual ref line
                        lineBuilder.AddLine(p1, p1 + axisDir * 0.3f);

                        // Draw Arc for Absolute Angle (Filled)
                        var pVectorEnd = p1 + Vector3.Normalize(p2 - p1) * 0.3f;
                        var pAxisEnd = p1 + axisDir * 0.3f;
                        var arcPoints = _analysisService.CalculateArcPoints(p1, pVectorEnd, pAxisEnd, 0.1f);
                        
                        if (arcPoints != null && arcPoints.Count > 1)
                        {
                            // Outline
                            for(int i=0; i<arcPoints.Count-1; i++)
                            {
                                lineBuilder.AddLine(arcPoints[i], arcPoints[i+1]);
                            }
                            // Fill (Triangle Fan)
                            for(int i=0; i<arcPoints.Count-1; i++)
                            {
                                meshBuilder.AddTriangle(p1, arcPoints[i], arcPoints[i+1]); // Vertex, P[i], P[i+1]
                                meshBuilder.AddTriangle(p1, arcPoints[i+1], arcPoints[i]); // Backface
                            }
                        }
                        
                        // Label mid-point
                        textInfo.TextInfo.Add(new TextInfo($"{angle.Value:F1}°", (p1 + p2) * 0.5f + new Vector3(0, 0.05f, 0)) 
                        { 
                            Foreground = Color4.White, Scale = 0.5f 
                        });
                    }
                }
            }
            else if (count == 3)
            {
                // Relative Angle
                int idx1 = AnalysisSelectedMarkers[0];
                int vertexIdx = AnalysisSelectedMarkers[1]; // Middle selected is vertex
                int idx3 = AnalysisSelectedMarkers[2];

                var p1 = motion.Markers.GetPosition(idx1, frame);
                var vertex = motion.Markers.GetPosition(vertexIdx, frame);
                var p3 = motion.Markers.GetPosition(idx3, frame);

                if (!float.IsNaN(p1.X) && !float.IsNaN(vertex.X) && !float.IsNaN(p3.X))
                {
                    lineBuilder.AddLine(p1, vertex);
                    lineBuilder.AddLine(vertex, p3);

                    float? angle = _analysisService.CalculateRelativeAngle(p1, vertex, p3);
                    
                    if (angle.HasValue)
                    {
                        resultText = $"Relative Angle\n{angle.Value:F1}°";

                        // Draw Arc (Filled)
                        var arcPoints = _analysisService.CalculateArcPoints(vertex, p1, p3, 0.1f);
                        if (arcPoints != null && arcPoints.Count > 1)
                        {
                            // Outline
                            for(int i=0; i<arcPoints.Count-1; i++)
                            {
                                lineBuilder.AddLine(arcPoints[i], arcPoints[i+1]);
                            }
                            // Fill (Triangle Fan)
                            for(int i=0; i<arcPoints.Count-1; i++)
                            {
                                meshBuilder.AddTriangle(vertex, arcPoints[i], arcPoints[i+1]); // Center, P[i], P[i+1]
                                meshBuilder.AddTriangle(vertex, arcPoints[i+1], arcPoints[i]); // Backface
                            }
                        }
                        
                        // Label at vertex
                        textInfo.TextInfo.Add(new TextInfo($"{angle.Value:F1}°", vertex + new Vector3(0, 0.05f, 0)) 
                        { 
                            Foreground = Color4.White, Scale = 0.5f 
                        });
                    }
                }
            }
            else if (count == 4)
            {
                // Twist Analysis (Transverse Plane Rotation)
                // [0-1]: Reference Vector (e.g., Pelvis)
                // [2-3]: Target Vector (e.g., Shoulders)
                
                int idx1 = AnalysisSelectedMarkers[0];
                int idx2 = AnalysisSelectedMarkers[1];
                int idx3 = AnalysisSelectedMarkers[2];
                int idx4 = AnalysisSelectedMarkers[3];

                var p1 = motion.Markers.GetPosition(idx1, frame);
                var p2 = motion.Markers.GetPosition(idx2, frame);
                var p3 = motion.Markers.GetPosition(idx3, frame);
                var p4 = motion.Markers.GetPosition(idx4, frame);

                if (!float.IsNaN(p1.X) && !float.IsNaN(p2.X) && !float.IsNaN(p3.X) && !float.IsNaN(p4.X))
                {
                    // Draw Vector Lines
                    lineBuilder.AddLine(p1, p2);
                    lineBuilder.AddLine(p3, p4);

                    // Calculate Angle
                    float? twistAngle = _analysisService.CalculateTwistAngle(p1, p2, p3, p4);

                    if (twistAngle.HasValue)
                    {
                        resultText = $"Twist Angle\n{twistAngle.Value:+0.0;-0.0}°";

                        // --- Visualization: Holographic Compass ---
                        
                        // Colors
                        var colorRef = new Color4(0.0f, 1.0f, 1.0f, 1.0f); // Cyan
                        var colorTrg = new Color4(1.0f, 0.5f, 0.0f, 1.0f); // Orange
                        var colorArc = new Color4(1.0f, 1.0f, 0.0f, 0.4f); // Yellow Ribbon

                        // Center of the visualization (Average Position)
                        Vector3 center = (p1 + p2 + p3 + p4) / 4.0f;
                        float radius = 0.4f;
                        float thickness = 0.05f; 

                        // 1. Floating Disc Ring (Visual Base)
                        int circleSegments = 64;
                        for (int i = 0; i < circleSegments; i++)
                        {
                            float angle = i * (2 * MathF.PI) / circleSegments;
                            float nextAngle = (i + 1) * (2 * MathF.PI) / circleSegments;
                            
                            Vector3 c1 = center + new Vector3(MathF.Cos(angle), 0, MathF.Sin(angle)) * radius;
                            Vector3 c2 = center + new Vector3(MathF.Cos(nextAngle), 0, MathF.Sin(nextAngle)) * radius;
                            
                            lineBuilder.AddLine(c1, c2); 
                        }

                        // 2. Projected Vectors (Centered on Disc)
                        // Project vectors to flat plane and normalize
                        Vector3 vRef = p2 - p1; vRef.Y = 0; 
                        if (vRef.LengthSquared() > 0) vRef = Vector3.Normalize(vRef);
                        
                        Vector3 vTar = p4 - p3; vTar.Y = 0; 
                        if (vTar.LengthSquared() > 0) vTar = Vector3.Normalize(vTar);

                        Vector3 centerRefEnd = center + vRef * radius;
                        Vector3 centerTarEnd = center + vTar * radius;

                        // Draw Arrows on Disc
                        // Ref Arrow (Cyan)
                        lineBuilder.AddLine(center, centerRefEnd);
                        // Arrowhead logic could be added here, stick to lines for now or use specific geometry if needed
                        // Just drawing a thicker recognizable line structure:
                        lineBuilder.AddLine(centerRefEnd, centerRefEnd - vRef * 0.05f + Vector3.Cross(vRef, Vector3.UnitY) * 0.02f);
                        lineBuilder.AddLine(centerRefEnd, centerRefEnd - vRef * 0.05f - Vector3.Cross(vRef, Vector3.UnitY) * 0.02f);

                        // Target Arrow (Orange)
                        lineBuilder.AddLine(center, centerTarEnd);
                        lineBuilder.AddLine(centerTarEnd, centerTarEnd - vTar * 0.05f + Vector3.Cross(vTar, Vector3.UnitY) * 0.02f);
                        lineBuilder.AddLine(centerTarEnd, centerTarEnd - vTar * 0.05f - Vector3.Cross(vTar, Vector3.UnitY) * 0.02f);

                        // 3. Drop Lines (Vertical Projection Indicators)
                        // Show where the vectors come from
                        // Make these distinct colors
                        lineBuilder.AddLine(p1, p2); // Source Body Line
                        lineBuilder.AddLine(p3, p4); // Target Body Line
                        
                        // Dotted projection effect (simulated by segments? or just thin lines)
                        // We'll just use thin lines from the actual markers to the disc plane
                        float discY = center.Y;
                        lineBuilder.AddLine(p1, new Vector3(p1.X, discY, p1.Z));
                        lineBuilder.AddLine(p2, new Vector3(p2.X, discY, p2.Z));
                        lineBuilder.AddLine(p3, new Vector3(p3.X, discY, p3.Z));
                        lineBuilder.AddLine(p4, new Vector3(p4.X, discY, p4.Z));

                        // 4. Ribbon Arc (Filled 3D Sector)
                        float startAng = MathF.Atan2(vRef.Z, vRef.X);
                        float sweepAng = twistAngle.Value * (MathF.PI / 180.0f);
                        int segments = 30;

                        var topCenter = center + Vector3.UnitY * (thickness / 2);
                        var botCenter = center - Vector3.UnitY * (thickness / 2);

                        for (int i = 0; i < segments; i++)
                        {
                            float ratio = (float)i / segments;
                            float nextRatio = (float)(i + 1) / segments;

                            float currentRad = startAng + sweepAng * ratio;
                            float nextRad = startAng + sweepAng * nextRatio;

                            Vector3 dir1 = new Vector3(MathF.Cos(currentRad), 0, MathF.Sin(currentRad));
                            Vector3 dir2 = new Vector3(MathF.Cos(nextRad), 0, MathF.Sin(nextRad));

                            Vector3 pTop1 = topCenter + dir1 * radius;
                            Vector3 pTop2 = topCenter + dir2 * radius;
                            Vector3 pBot1 = botCenter + dir1 * radius;
                            Vector3 pBot2 = botCenter + dir2 * radius;

                            // Fill
                            meshBuilder.AddTriangle(topCenter, pTop1, pTop2);
                            meshBuilder.AddTriangle(topCenter, pTop2, pTop1);
                            meshBuilder.AddTriangle(botCenter, pBot2, pBot1); 
                            meshBuilder.AddTriangle(botCenter, pBot1, pBot2);
                            meshBuilder.AddTriangle(pTop1, pBot1, pBot2);
                            meshBuilder.AddTriangle(pBot2, pTop2, pTop1);
                            meshBuilder.AddTriangle(pTop1, pBot2, pBot1);
                            meshBuilder.AddTriangle(pBot2, pTop1, pTop2);
                        }

                        // Labels
                        textInfo.TextInfo.Add(new TextInfo("Ref", centerRefEnd + Vector3.UnitY * 0.05f) { Foreground = colorRef, Scale = 0.4f });
                        textInfo.TextInfo.Add(new TextInfo("Trg", centerTarEnd + Vector3.UnitY * 0.05f) { Foreground = colorTrg, Scale = 0.4f });
                        textInfo.TextInfo.Add(new TextInfo($"{twistAngle.Value:+0.0;-0.0}°", center + Vector3.UnitY * (thickness + 0.1f)) 
                        { 
                            Foreground = Color4.White, Scale = 0.6f 
                        });
                    }
                }
            }

            AnalysisResultText = string.IsNullOrEmpty(resultText) ? "Select markers..." : resultText;

            // Update Geometry
            if (_analysisLinesModel != null)
            {
                _analysisLinesModel.Geometry = lineBuilder.ToLineGeometry3D();
                _analysisLinesModel.IsRendering = true;
            }
            if (_analysisArcMeshModel != null)
            {
                _analysisArcMeshModel.Geometry = meshBuilder.ToMeshGeometry3D();
                _analysisArcMeshModel.IsRendering = true;
            }
            if (_analysisLabelModel != null)
            {
                _analysisLabelModel.Geometry = textInfo;
                _analysisLabelModel.IsRendering = true;
            }
        }
    }
}
