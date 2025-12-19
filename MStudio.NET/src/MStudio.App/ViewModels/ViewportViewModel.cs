using System;
using System.Collections.ObjectModel;
using System.Windows.Media.Media3D;
using System.Numerics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HelixToolkit;
using HelixToolkit.Geometry;
using HelixToolkit.Maths;
using HelixToolkit.SharpDX;
using HelixToolkit.Wpf.SharpDX;
using MStudio.Core.Models;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    public partial class MStudioViewportViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;

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

        // Trajectory & Bones
        private LineGeometryModel3D? _trajectoryModel;
        private LineGeometryModel3D? _boneModel;

        [ObservableProperty]
        private int _selectedMarkerIndex = -1;

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

        // Appearance settings
        private const float MarkerRadius = 0.012f;
        private readonly Color4 _markerColor = new Color4(0.2f, 0.8f, 0.3f, 1.0f); // Green
        private readonly System.Windows.Media.Color _trajectoryColor = System.Windows.Media.Colors.Yellow;
        private readonly System.Windows.Media.Color _boneColor = System.Windows.Media.Color.FromArgb(180, 200, 200, 200); // White-grey

        public MStudioViewportViewModel(ISessionService sessionService, ITimelineService timelineService)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;

            EffectsManager = new DefaultEffectsManager();

            var meshBuilder = new MeshBuilder(true, false);
            meshBuilder.AddSphere(new Vector3(0, 0, 0), MarkerRadius, 12, 12);
            _markerSphereGeometry = meshBuilder.ToMeshGeometry3D();

            Camera = new HelixToolkit.Wpf.SharpDX.PerspectiveCamera
            {
                Position = new Point3D(2, 2, 2),
                LookDirection = new Vector3D(-2, -2, -2),
                UpDirection = new Vector3D(0, 1, 0),
                FarPlaneDistance = 1000,
                NearPlaneDistance = 0.01
            };

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
        }

        private void CreateTrajectoryAndBoneModels()
        {
            _trajectoryModel = new LineGeometryModel3D
            {
                Color = _trajectoryColor,
                Thickness = 1.0,
                IsRendering = true
            };
            SceneElements.Add(_trajectoryModel);

            _boneModel = new LineGeometryModel3D
            {
                Color = _boneColor,
                Thickness = 1.5,
                IsRendering = true
            };
            SceneElements.Add(_boneModel);
        }

        partial void OnSelectedMarkerIndexChanged(int value)
        {
            UpdateTrajectories();
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
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = zBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.SkyBlue, Thickness = 2 });
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
                _originModel.Children.Add(new LineGeometryModel3D { Geometry = zBuilder.ToLineGeometry3D(), Color = System.Windows.Media.Colors.SkyBlue, Thickness = 2 });
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

            AutoGenerateBones(motion);
            UpdateMarkersAndVisuals();

            MarkerCountText = $"Markers: {motion.Markers.MarkerCount}";
        }

        private void AutoGenerateBones(MotionData motion)
        {
            // Simple proximity and name-based auto-skeleton
            // In a production app, this would use a template (e.g. MarkerSets)
            var names = motion.Metadata.MarkerNames;
            for (int i = 0; i < names.Count; i++)
            {
                for (int j = i + 1; j < names.Count; j++)
                {
                    var p1 = motion.Markers.GetPosition(i, 0);
                    var p2 = motion.Markers.GetPosition(j, 0);

                    if (float.IsNaN(p1.X) || float.IsNaN(p2.X)) continue;

                    float dist = Vector3.Distance(p1, p2);
                    // Connect if markers are close (e.g. within 30cm) and probably on same limb
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
    }
}
