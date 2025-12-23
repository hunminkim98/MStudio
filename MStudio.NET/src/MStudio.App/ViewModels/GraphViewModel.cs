using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using CommunityToolkit.Mvvm.ComponentModel;
using MStudio.Services.Interfaces;

namespace MStudio.App.ViewModels
{
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

        [ObservableProperty]
        private double _viewWidth = 800;

        [ObservableProperty]
        private double _viewHeight = 100;

        [ObservableProperty]
        private double _cursorPosition = 0;

        // Y-axis range labels
        [ObservableProperty]
        private string _yAxisMax = "";

        [ObservableProperty]
        private string _yAxisMid = "";

        [ObservableProperty]
        private string _yAxisMin = "";

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
        }

        public void UpdatePoints()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || SelectedMarkerIndex < 0 || motion.Markers.MarkerCount <= SelectedMarkerIndex)
            {
                PointsX = new();
                PointsY = new();
                PointsZ = new();
                return;
            }

            int frameCount = motion.Markers.FrameCount;
            if (frameCount <= 0) return;

            // First pass: Find actual data range for all XYZ values
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            
            var validPositions = new List<(int frame, System.Numerics.Vector3 pos)>();
            
            for (int f = 0; f < frameCount; f++)
            {
                var pos = motion.Markers.GetPosition(SelectedMarkerIndex, f);
                
                if (!float.IsNaN(pos.X) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                {
                    validPositions.Add((f, pos));
                    
                    minVal = Math.Min(minVal, Math.Min(pos.X, Math.Min(pos.Y, pos.Z)));
                    maxVal = Math.Max(maxVal, Math.Max(pos.X, Math.Max(pos.Y, pos.Z)));
                }
            }

            if (validPositions.Count == 0)
            {
                PointsX = new();
                PointsY = new();
                PointsZ = new();
                return;
            }

            // Add 5% padding to the range so data doesn't touch edges
            float range = maxVal - minVal;
            if (range < 0.001f) range = 1.0f; // Prevent division by zero for constant data
            
            float padding = range * 0.05f;
            minVal -= padding;
            maxVal += padding;
            range = maxVal - minVal;

            // Second pass: Create point collections with proper scaling
            var px = new PointCollection();
            var py = new PointCollection();
            var pz = new PointCollection();

            double stepX = ViewWidth / Math.Max(1, frameCount - 1);

            foreach (var (frame, pos) in validPositions)
            {
                double x = frame * stepX;
                
                // Map values to pixel space (Y is inverted in WPF)
                px.Add(new Point(x, ViewHeight * (1 - (pos.X - minVal) / range)));
                py.Add(new Point(x, ViewHeight * (1 - (pos.Y - minVal) / range)));
                pz.Add(new Point(x, ViewHeight * (1 - (pos.Z - minVal) / range)));
            }

            PointsX = px;
            PointsY = py;
            PointsZ = pz;

            // Update Y-axis labels
            YAxisMax = maxVal.ToString("F2");
            YAxisMid = ((minVal + maxVal) / 2).ToString("F2");
            YAxisMin = minVal.ToString("F2");

            UpdateCursor();
        }

        private void UpdateCursor()
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.FrameCount <= 0)
            {
                CursorPosition = 0;
                return;
            }

            double stepX = ViewWidth / Math.Max(1, motion.Markers.FrameCount - 1);
            CursorPosition = _timelineService.CurrentFrame * stepX;
        }

        /// <summary>
        /// Seeks to a frame based on X position in the view.
        /// Called when user clicks or drags on the graph panel.
        /// </summary>
        public void SeekToPosition(double xPosition)
        {
            var motion = _sessionService.CurrentMotion;
            if (motion == null || motion.Markers.FrameCount <= 0 || ViewWidth <= 0)
                return;

            int totalFrames = motion.Markers.FrameCount;
            
            // Convert X position to frame
            double ratio = Math.Clamp(xPosition / ViewWidth, 0, 1);
            int targetFrame = (int)Math.Round(ratio * (totalFrames - 1));
            
            // Set the frame via timeline service
            _timelineService.CurrentFrame = targetFrame;
        }
    }
}
