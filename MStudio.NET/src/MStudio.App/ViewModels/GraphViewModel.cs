using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using MStudio.Core.Messaging;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
    /// <summary>
    /// Display modes for the graph view.
    /// </summary>
    public enum GraphDisplayMode
    {
        Combined,    // All XYZ in one graph
        Separation,  // X, Y, Z in separate columns
        X,           // X only
        Y,           // Y only
        Z            // Z only
    }

    public partial class GraphViewModel : ObservableObject
    {
        private readonly ISessionService _sessionService;
        private readonly ITimelineService _timelineService;

        [ObservableProperty]
        private int _selectedMarkerIndex = -1;

        [ObservableProperty]
        private PointCollection _pointsX = new();

        [ObservableProperty]
        private PointCollection _pointsY = new();

        [ObservableProperty]
        private PointCollection _pointsZ = new();

        // Separation mode points (scaled to column width)
        [ObservableProperty]
        private PointCollection _pointsX_Sep = new();

        [ObservableProperty]
        private PointCollection _pointsY_Sep = new();

        [ObservableProperty]
        private PointCollection _pointsZ_Sep = new();

        [ObservableProperty]
        private double _viewWidth = 800;

        [ObservableProperty]
        private double _viewHeight = 100;

        [ObservableProperty]
        private double _cursorPosition = 0;

        // Cursor position for separation mode (scaled to column width)
        [ObservableProperty]
        private double _cursorPosition_Sep = 0;

        // Y-axis range labels
        [ObservableProperty]
        private string _yAxisMax = "";

        [ObservableProperty]
        private string _yAxisMid = "";

        [ObservableProperty]
        private string _yAxisMin = "";

        // Separation Mode Y-axis labels
        [ObservableProperty]
        private string _yAxisMax_SepX = "";
        [ObservableProperty]
        private string _yAxisMin_SepX = "";

        [ObservableProperty]
        private string _yAxisMax_SepY = "";
        [ObservableProperty]
        private string _yAxisMin_SepY = "";

        [ObservableProperty]
        private string _yAxisMax_SepZ = "";
        [ObservableProperty]
        private string _yAxisMin_SepZ = "";

        // Graph Display Mode
        [ObservableProperty]
        private GraphDisplayMode _selectedDisplayMode = GraphDisplayMode.Combined;

        /// <summary>
        /// Available display modes for the dropdown.
        /// </summary>
        public ObservableCollection<GraphDisplayMode> DisplayModes { get; } = new()
        {
            GraphDisplayMode.Combined,
            GraphDisplayMode.Separation,
            GraphDisplayMode.X,
            GraphDisplayMode.Y,
            GraphDisplayMode.Z
        };

        /// <summary>
        /// Called when display mode changes to update visibility properties.
        /// </summary>
        partial void OnSelectedDisplayModeChanged(GraphDisplayMode value)
        {
            OnPropertyChanged(nameof(ShowXCurve));
            OnPropertyChanged(nameof(ShowYCurve));
            OnPropertyChanged(nameof(ShowZCurve));
            OnPropertyChanged(nameof(IsSeparationMode));
            OnPropertyChanged(nameof(IsCombinedOrSingleMode));
            
            // Re-calculate points with new scaling rules
            UpdatePoints();
        }

        /// <summary>
        /// Whether to show X curve in the combined/single graph area.
        /// </summary>
        public bool ShowXCurve => SelectedDisplayMode == GraphDisplayMode.Combined || 
                                   SelectedDisplayMode == GraphDisplayMode.X;

        /// <summary>
        /// Whether to show Y curve in the combined/single graph area.
        /// </summary>
        public bool ShowYCurve => SelectedDisplayMode == GraphDisplayMode.Combined || 
                                   SelectedDisplayMode == GraphDisplayMode.Y;

        /// <summary>
        /// Whether to show Z curve in the combined/single graph area.
        /// </summary>
        public bool ShowZCurve => SelectedDisplayMode == GraphDisplayMode.Combined || 
                                   SelectedDisplayMode == GraphDisplayMode.Z;

        /// <summary>
        /// Whether the graph is in Separation mode (3 columns).
        /// </summary>
        public bool IsSeparationMode => SelectedDisplayMode == GraphDisplayMode.Separation;

        /// <summary>
        /// Whether the graph is in Combined or single-axis mode.
        /// </summary>
        public bool IsCombinedOrSingleMode => SelectedDisplayMode != GraphDisplayMode.Separation;

        // Value Visibility
        [ObservableProperty]
        private bool _isValueVisible = false;

        partial void OnIsValueVisibleChanged(bool value)
        {
            if (value) UpdateCursor();
        }

        [ObservableProperty]
        private string _currentXValue = "";

        [ObservableProperty]
        private string _currentYValue = "";

        [ObservableProperty]
        private string _currentZValue = "";

        // Edit Mode properties
        [ObservableProperty]
        private bool _isEditModeEnabled = false;

        /// <summary>
        /// Called when IsEditModeEnabled changes.
        /// Clears selection when edit mode is disabled.
        /// </summary>
        partial void OnIsEditModeEnabledChanged(bool value)
        {
            if (!value)
            {
                // Clear selection when edit mode is turned off
                ClearSelection();
            }
        }

        // Range Selection properties
        [ObservableProperty]
        private double _selectionStartX = 0;

        [ObservableProperty]
        private double _selectionEndX = 0;

        [ObservableProperty]
        private bool _isSelecting = false;

        [ObservableProperty]
        private Visibility _selectionVisibility = Visibility.Collapsed;

        [ObservableProperty]
        private int _selectionStartFrame = 0;

        [ObservableProperty]
        private int _selectionEndFrame = 0;

        /// <summary>
        /// Gets the width of the selection rectangle.
        /// </summary>
        public double SelectionWidth => Math.Abs(SelectionEndX - SelectionStartX);

        /// <summary>
        /// Gets the left position of the selection rectangle (always the smaller X value).
        /// </summary>
        public double SelectionLeft => Math.Min(SelectionStartX, SelectionEndX);

        /// <summary>
        /// Gets the width of the selection rectangle for Separation mode (scaled to column width).
        /// </summary>
        public double SelectionWidth_Sep => SelectionWidth / 3.0;

        /// <summary>
        /// Gets the left position of the selection rectangle for Separation mode.
        /// </summary>
        public double SelectionLeft_Sep => SelectionLeft / 3.0;

        public GraphViewModel(ISessionService sessionService, ITimelineService timelineService)
        {
            _sessionService = sessionService;
            _timelineService = timelineService;

            _sessionService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ISessionService.CurrentMotion))
                {
                    UpdatePoints();
                }
            };

            _timelineService.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ITimelineService.CurrentFrame))
                {
                    UpdateCursor();
                }
            };

            PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(SelectedMarkerIndex))
                {
                    UpdatePoints();
                }
            };

            // Register for marker data change messages (from MainViewModel after FillGaps/Smooth)
            WeakReferenceMessenger.Default.Register<MarkerDataChangedMessage>(this, (r, m) =>
            {
                UpdatePoints();
            });

            // Register for marker selection sync (from other ViewModels)
            WeakReferenceMessenger.Default.Register<MarkerSelectionChangedMessage>(this, (r, m) =>
            {
                // Avoid infinite loop - don't update if we are the source
                if (m.Source != this)
                {
                    SelectedMarkerIndex = m.SelectedMarkerIndex;
                }
            });
        }

        #region Property Change Handlers (for SizeObserverBehavior)

        /// <summary>
        /// Called when ViewWidth changes (via SizeObserverBehavior binding).
        /// Triggers graph re-rendering with new dimensions.
        /// </summary>
        partial void OnViewWidthChanged(double value)
        {
            if (value > 0)
            {
                UpdatePoints();
            }
        }

        /// <summary>
        /// Called when ViewHeight changes (via SizeObserverBehavior binding).
        /// Triggers graph re-rendering with new dimensions.
        /// </summary>
        partial void OnViewHeightChanged(double value)
        {
            if (value > 0)
            {
                UpdatePoints();
            }
        }

        #endregion

        #region Commands (for MouseSeekBehavior)

        /// <summary>
        /// Command for seeking to a position. Used by MouseSeekBehavior.
        /// Parameter is the X position in pixels.
        /// Left click always seeks - right click is used for range selection.
        /// </summary>
        [RelayCommand]
        private void Seek(double xPosition)
        {
            SeekToPosition(xPosition);
        }

        #endregion

        #region Range Selection Commands

        /// <summary>
        /// Converts X position from column-local to full ViewWidth scale.
        /// In Separation mode, each column is 1/3 of ViewWidth, so we multiply by 3.
        /// </summary>
        private double ScaleXPositionForSelection(double xPosition)
        {
            if (IsSeparationMode)
            {
                // In separation mode, xPosition is within a column (0 to ViewWidth/3)
                // We need to scale it to full ViewWidth range
                return xPosition * 3.0;
            }
            return xPosition;
        }

        /// <summary>
        /// Starts a new range selection at the given X position.
        /// </summary>
        [RelayCommand]
        private void StartSelection(double xPosition)
        {
            if (!IsEditModeEnabled) return;

            double scaledX = ScaleXPositionForSelection(xPosition);

            IsSelecting = true;
            SelectionStartX = scaledX;
            SelectionEndX = scaledX;
            SelectionVisibility = Visibility.Visible;
            
            UpdateSelectionFrames();
            NotifySelectionPropertiesChanged();
        }

        /// <summary>
        /// Updates the selection end position during drag.
        /// </summary>
        [RelayCommand]
        private void UpdateSelection(double xPosition)
        {
            if (!IsEditModeEnabled || !IsSelecting) return;

            double scaledX = ScaleXPositionForSelection(xPosition);
            SelectionEndX = Math.Clamp(scaledX, 0, ViewWidth);
            UpdateSelectionFrames();
            NotifySelectionPropertiesChanged();
        }

        /// <summary>
        /// Ends the current selection.
        /// </summary>
        [RelayCommand]
        private void EndSelection()
        {
            if (!IsEditModeEnabled) return;

            IsSelecting = false;
            
            // Ensure start < end for consistent frame range
            if (SelectionStartX > SelectionEndX)
            {
                (SelectionStartX, SelectionEndX) = (SelectionEndX, SelectionStartX);
            }
            
            UpdateSelectionFrames();
            NotifySelectionPropertiesChanged();
        }

        /// <summary>
        /// Clears the current selection.
        /// </summary>
        [RelayCommand]
        private void ClearSelection()
        {
            SelectionStartX = 0;
            SelectionEndX = 0;
            SelectionStartFrame = 0;
            SelectionEndFrame = 0;
            SelectionVisibility = Visibility.Collapsed;
            IsSelecting = false;
            NotifySelectionPropertiesChanged();
        }

        /// <summary>
        /// Updates the frame numbers based on current selection X positions.
        /// </summary>
        private void UpdateSelectionFrames()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.FrameCount <= 0 || ViewWidth <= 0)
            {
                SelectionStartFrame = 0;
                SelectionEndFrame = 0;
                return;
            }

            int totalFrames = motion.Markers.FrameCount;
            
            double startRatio = Math.Clamp(Math.Min(SelectionStartX, SelectionEndX) / ViewWidth, 0, 1);
            double endRatio = Math.Clamp(Math.Max(SelectionStartX, SelectionEndX) / ViewWidth, 0, 1);
            
            SelectionStartFrame = (int)Math.Round(startRatio * (totalFrames - 1));
            SelectionEndFrame = (int)Math.Round(endRatio * (totalFrames - 1));
        }

        /// <summary>
        /// Notifies that selection-related computed properties have changed.
        /// </summary>
        private void NotifySelectionPropertiesChanged()
        {
            OnPropertyChanged(nameof(SelectionWidth));
            OnPropertyChanged(nameof(SelectionLeft));
            OnPropertyChanged(nameof(SelectionWidth_Sep));
            OnPropertyChanged(nameof(SelectionLeft_Sep));
        }

        #endregion

        /// <summary>
        /// Update points and scaling based on current mode.
        /// </summary>
        public void UpdatePoints()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || SelectedMarkerIndex < 0 || motion.Markers.MarkerCount <= SelectedMarkerIndex)
            {
                ClearPoints();
                return;
            }

            int frameCount = motion.Markers.FrameCount;
            if (frameCount <= 0) return;

            // First pass: Find actual data range for X, Y, Z individually
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            float minZ = float.MaxValue, maxZ = float.MinValue;
            
            var validPositions = new List<(int frame, System.Numerics.Vector3 pos)>();
            
            for (int f = 0; f < frameCount; f++)
            {
                var pos = motion.Markers.GetPosition(SelectedMarkerIndex, f);
                
                if (!float.IsNaN(pos.X) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                {
                    validPositions.Add((f, pos));
                    
                    if (pos.X < minX) minX = pos.X;
                    if (pos.X > maxX) maxX = pos.X;
                    
                    if (pos.Y < minY) minY = pos.Y;
                    if (pos.Y > maxY) maxY = pos.Y;
                    
                    if (pos.Z < minZ) minZ = pos.Z;
                    if (pos.Z > maxZ) maxZ = pos.Z;
                }
            }

            if (validPositions.Count == 0)
            {
                ClearPoints();
                return;
            }

            // Calculate ranges with padding
            (float minX_p, float rangeX) = CalculateRange(minX, maxX);
            (float minY_p, float rangeY) = CalculateRange(minY, maxY);
            (float minZ_p, float rangeZ) = CalculateRange(minZ, maxZ);

            // Combined range (min of all mins, max of all maxs)
            float minCombined = Math.Min(minX, Math.Min(minY, minZ));
            float maxCombined = Math.Max(maxX, Math.Max(maxY, maxZ));
            (float minCombined_p, float rangeCombined) = CalculateRange(minCombined, maxCombined);

            // Determine which range to use for main graph based on mode
            float mainMin, mainRange;
            switch (SelectedDisplayMode)
            {
                case GraphDisplayMode.X:
                    mainMin = minX_p; mainRange = rangeX;
                    break;
                case GraphDisplayMode.Y:
                    mainMin = minY_p; mainRange = rangeY;
                    break;
                case GraphDisplayMode.Z:
                    mainMin = minZ_p; mainRange = rangeZ;
                    break;
                case GraphDisplayMode.Combined:
                default: 
                    mainMin = minCombined_p; mainRange = rangeCombined;
                    break;
            }

            // Create point collections
            var px = new PointCollection();
            var py = new PointCollection();
            var pz = new PointCollection();

            double stepX = ViewWidth / Math.Max(1, frameCount - 1);

            foreach (var (frame, pos) in validPositions)
            {
                double x = frame * stepX;
                
                // Map values to pixel space using selected main range
                px.Add(new Point(x, ScaleValue(pos.X, mainMin, mainRange, ViewHeight)));
                py.Add(new Point(x, ScaleValue(pos.Y, mainMin, mainRange, ViewHeight)));
                pz.Add(new Point(x, ScaleValue(pos.Z, mainMin, mainRange, ViewHeight)));
            }

            PointsX = px;
            PointsY = py;
            PointsZ = pz;

            // Handle Separation Mode Points (each uses its own range)
            double columnWidth = ViewWidth / 3.0; // Assuming 3 columns
            double stepX_Sep = columnWidth / Math.Max(1, frameCount - 1);

            var px_sep = new PointCollection();
            var py_sep = new PointCollection();
            var pz_sep = new PointCollection();

            foreach (var (frame, pos) in validPositions)
            {
                double x_sep = frame * stepX_Sep;
                
                // Use individual ranges for separation mode
                px_sep.Add(new Point(x_sep, ScaleValue(pos.X, minX_p, rangeX, ViewHeight)));
                py_sep.Add(new Point(x_sep, ScaleValue(pos.Y, minY_p, rangeY, ViewHeight)));
                pz_sep.Add(new Point(x_sep, ScaleValue(pos.Z, minZ_p, rangeZ, ViewHeight)));
            }

            PointsX_Sep = px_sep;
            PointsY_Sep = py_sep;
            PointsZ_Sep = pz_sep;

            // Update Y-axis labels based on MAIN range (Combined/X/Y/Z)
            YAxisMax = (mainMin + mainRange).ToString("F2");
            YAxisMid = (mainMin + (mainRange / 2)).ToString("F2");
            YAxisMin = mainMin.ToString("F2");

            // Update Separation Mode Y-axis labels
            YAxisMax_SepX = (minX_p + rangeX).ToString("F2");
            YAxisMin_SepX = minX_p.ToString("F2");

            YAxisMax_SepY = (minY_p + rangeY).ToString("F2");
            YAxisMin_SepY = minY_p.ToString("F2");

            YAxisMax_SepZ = (minZ_p + rangeZ).ToString("F2");
            YAxisMin_SepZ = minZ_p.ToString("F2");

            UpdateCursor();
        }

        private void ClearPoints()
        {
            PointsX = new();
            PointsY = new();
            PointsZ = new();
            PointsX_Sep = new();
            PointsY_Sep = new();
            PointsZ_Sep = new();
        }

        private (float min, float range) CalculateRange(float min, float max)
        {
            float range = max - min;
            if (range < 0.001f) range = 1.0f;
            float padding = range * 0.05f;
            return (min - padding, range + 2 * padding);
        }

        private double ScaleValue(float value, float min, float range, double height)
        {
            return height * (1 - (value - min) / range);
        }

        private void UpdateCursor()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.FrameCount <= 0)
            {
                CursorPosition = 0;
                CursorPosition_Sep = 0;
                CurrentXValue = ""; CurrentYValue = ""; CurrentZValue = "";
                return;
            }

            double stepX = ViewWidth / Math.Max(1, motion.Markers.FrameCount - 1);
            CursorPosition = _timelineService.CurrentFrame * stepX;

            // Separation mode cursor (scaled to column width)
            double columnWidth = ViewWidth / 3.0;
            double stepX_Sep = columnWidth / Math.Max(1, motion.Markers.FrameCount - 1);
            CursorPosition_Sep = _timelineService.CurrentFrame * stepX_Sep;

            // Update current values if visible
            if (IsValueVisible && SelectedMarkerIndex >= 0 && SelectedMarkerIndex < motion.Markers.MarkerCount)
            {
                int frame = _timelineService.CurrentFrame;
                if (frame >= 0 && frame < motion.Markers.FrameCount)
                {
                    var pos = motion.Markers.GetPosition(SelectedMarkerIndex, frame);
                    
                    if (!float.IsNaN(pos.X) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    {
                        CurrentXValue = pos.X.ToString("F2");
                        CurrentYValue = pos.Y.ToString("F2");
                        CurrentZValue = pos.Z.ToString("F2");
                    }
                    else
                    {
                        CurrentXValue = ""; CurrentYValue = ""; CurrentZValue = "";
                    }
                }
            }
        }

        /// <summary>
        /// Seeks to a frame based on X position in the view.
        /// Called by SeekCommand when user clicks or drags on the graph panel.
        /// </summary>
        private void SeekToPosition(double xPosition)
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.FrameCount <= 0 || ViewWidth <= 0)
                return;

            // Scale X position for Separation mode
            double scaledX = ScaleXPositionForSelection(xPosition);

            int totalFrames = motion.Markers.FrameCount;
            
            // Convert X position to frame
            double ratio = Math.Clamp(scaledX / ViewWidth, 0, 1);
            int targetFrame = (int)Math.Round(ratio * (totalFrames - 1));
            
            // Set the frame via timeline service
            _timelineService.CurrentFrame = targetFrame;
        }
    }
}

