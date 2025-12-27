using System;
using System.Collections.ObjectModel;
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
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    public partial class MStudioViewportViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;
        private readonly IVisualizationSettingsService _visualizationSettings;

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
        private bool _isShowTrajectory = true;

        partial void OnIsShowTrajectoryChanged(bool value)
        {
            if (_trajectoryModel != null) _trajectoryModel.IsRendering = value;
        }

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

        partial void OnIsZUpChanged(bool value)
        {
            // Re-create Grid and Axis to match orientation
            if (_gridModel != null) SceneElements.Remove(_gridModel);
            if (_originModel != null) SceneElements.Remove(_originModel);
            
            CreateGrid();
            CreateAxis();

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

        // Marker Rendering
        private InstancingMeshGeometryModel3D? _markerModel;
        private HelixToolkit.SharpDX.MeshGeometry3D? _markerSphereGeometry;

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
            IVisualizationSettingsService visualizationSettings)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;
            _visualizationSettings = visualizationSettings;

            // Sync initial values from service
            SyncFromVisualizationSettings();

            EffectsManager = new DefaultEffectsManager();

            var meshBuilder = new MeshBuilder(true, false);
            meshBuilder.AddSphere(new Vector3(0, 0, 0), MarkerSize, 12, 12);
            _markerSphereGeometry = meshBuilder.ToMeshGeometry3D();

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
            CreateTrajectoryAndBoneModels();

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
            _markerSize = _visualizationSettings.MarkerSize;
            _markerOpacity = _visualizationSettings.MarkerOpacity;
            _markerColor = new Color4(mc.R, mc.G, mc.B, _markerOpacity);
            
            var smc = _visualizationSettings.SelectedMarkerColor;
            _selectedMarkerColor = System.Windows.Media.Color.FromScRgb(smc.A, smc.R, smc.G, smc.B);
            
            _boneThickness = _visualizationSettings.BoneThickness;
            _boneOpacity = _visualizationSettings.BoneOpacity;
            var bc = _visualizationSettings.BoneColor;
            _boneColor = System.Windows.Media.Color.FromScRgb(_boneOpacity, bc.R, bc.G, bc.B);
            _isShowMarkerNames = _visualizationSettings.ShowMarkerNames;
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

            if (_markerModel != null)
            {
                _markerModel.Geometry = _markerSphereGeometry;
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

        partial void OnSelectedMarkerIndexChanged(int value)
        {
            UpdateTrajectories();

            // Publish selection change message for other ViewModels to sync
            WeakReferenceMessenger.Default.Send(new MarkerSelectionChangedMessage
            {
                SelectedMarkerIndex = value,
                Source = this
            });
        }

        private void CreateGrid()
        {
            var builder = new LineBuilder();
            int gridSize = 15;
            float gridSpacing = 0.3f; // Slightly wider grid spacing

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
            float axisLength = 0.3f; 
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

        private void OnMotionLoaded()
        {
            if (_markerModel != null) _markerModel.Instances = null;
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
                    Material = new HelixToolkit.Wpf.SharpDX.DiffuseMaterial { DiffuseColor = _markerColor }
                };
                SceneElements.Add(_markerModel);
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
            UpdateMarkerNames();
            UpdateTrajectories();
            UpdateBones();
        }

        public void UpdateMarkerPositions()
        {
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
        /// Updates the billboard text labels showing marker names in 3D space.
        /// </summary>
        public void UpdateMarkerNames()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || _markerNamesModel == null) return;

            int frame = _timelineService.CurrentFrame;
            if (frame < 0 || frame >= motion.Markers.FrameCount)
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

        public void UpdateTrajectories()
        {
            var motion = _sessionService.CurrentMotion;
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
            var motion = _sessionService.CurrentMotion;
            if (motion == null || _boneModel == null || _boneLinks.Count == 0)
            {
                if (_boneModel != null) _boneModel.Geometry = null;
                return;
            }

            int frame = _timelineService.CurrentFrame;
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
            var motion = _sessionService.CurrentMotion;
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
                SelectedMarkerIndex = bestMarkerIndex;
                return true;
            }

            return false;
        }
    }
}


